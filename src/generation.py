"""Stage 3: grounded answer generation.

Context building under a fixed token budget, prompt, robust JSON parsing,
two interchangeable LLM backends (local Hugging Face model or an
OpenAI-compatible API), and resumable runs saved to jsonl.
"""
import json
import re
import time
from pathlib import Path

# Prompt v2 (after the dev check): asks for a short phrase instead of "as briefly as possible",
# because v1 often copied whole sentences, which lowers token F1.
PROMPT_VERSION = "v2"

SYSTEM_PROMPT = (
    "You answer questions about a scientific paper. "
    "Use only the numbered excerpts you are given; do not use outside knowledge."
)

INSTRUCTIONS = (
    "Answer rules:\n"
    "- If the question is a yes/no question, answer exactly \"Yes\" or \"No\".\n"
    "- Otherwise answer with a short phrase, not a full sentence (usually fewer than 10 words), "
    "copying words from the excerpts when possible. If the answer is a list, separate the items with commas.\n"
    "- If the excerpts do not contain the answer, answer exactly \"Unanswerable\".\n"
    "- Always answer in English.\n"
    "Return only JSON: {\"answer\": \"...\", \"evidence\": [numbers of the excerpts you used]}"
)

CLOSED_BOOK_INSTRUCTIONS = (
    "Answer rules:\n"
    "- If the question is a yes/no question, answer exactly \"Yes\" or \"No\".\n"
    "- Otherwise answer with a short phrase, not a full sentence (usually fewer than 10 words). "
    "If the answer is a list, separate the items with commas.\n"
    "- If you do not know, answer exactly \"Unanswerable\".\n"
    "- Always answer in English.\n"
    "Return only JSON: {\"answer\": \"...\", \"evidence\": []}"
)


# ---------- context ----------

def build_context(chunk_ids, chunk_info, max_tokens=1280):
    """Take chunks in rank order until the token budget is reached.

    The same budget is used for every condition (retrievers and oracle), so
    differences in answer quality come from WHICH text is given, not how much.
    """
    selected, used = [], 0
    for cid in chunk_ids:
        n = chunk_info[cid]["n_tokens"]
        if used + n > max_tokens:
            break
        selected.append(cid)
        used += n
    return selected, used


def oracle_chunks(record, chunks_of_paper):
    """Chunks containing gold evidence, in document order (upper bound)."""
    gold = set(record["gold_paragraphs"])
    return [c["chunk_id"] for c in chunks_of_paper if gold & set(c["para_ids"])]


def make_prompt(question, title, context_ids, chunk_info):
    """Returns (messages, label_to_chunk_id). Excerpts are labelled [1], [2], ..."""
    if not context_ids:
        user = f"Paper title: {title}\n\nQuestion: {question}\n\n{CLOSED_BOOK_INSTRUCTIONS}"
        return [{"role": "system", "content": SYSTEM_PROMPT.replace(
                    "Use only the numbered excerpts you are given; do not use outside knowledge.",
                    "No excerpts are given; answer from what you know.")},
                {"role": "user", "content": user}], {}

    labels, blocks = {}, []
    for i, cid in enumerate(context_ids, start=1):
        c = chunk_info[cid]
        labels[i] = cid
        blocks.append(f"[{i}] (section: {c['section'] or 'n/a'})\n{c['text']}")
    user = (f"Paper title: {title}\n\nExcerpts:\n\n" + "\n\n".join(blocks)
            + f"\n\nQuestion: {question}\n\n{INSTRUCTIONS}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}], labels


# ---------- parsing ----------

def parse_output(raw, labels):
    """Extract {"answer", "evidence"} from the model output, tolerating
    code fences, extra text and invalid JSON. Evidence numbers are mapped
    back to chunk ids; unknown numbers are dropped."""
    answer, evidence, ok = None, [], False
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            answer = str(obj.get("answer", "")).strip()
            evidence = obj.get("evidence", []) or []
            ok = True
        except (json.JSONDecodeError, AttributeError):
            pass
    if answer is None:  # fallback: whole output is the answer
        answer = raw.strip().strip('"')
    cited = []
    for e in evidence if isinstance(evidence, list) else []:
        try:
            e = int(str(e).strip("[] "))
        except ValueError:
            continue
        if e in labels and labels[e] not in cited:
            cited.append(labels[e])
    return {"answer": answer, "cited_chunk_ids": cited, "parsed_json": ok}


# ---------- backends ----------

class HFBackend:
    """Local instruction model in Colab (4-bit to fit a free T4)."""

    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct", load_in_4bit=False, max_new_tokens=128):
        """load_in_4bit=True for small GPUs (Colab T4); otherwise bfloat16 (needs ~16 GB for a 7B model)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = {"device_map": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kwargs["dtype"] = torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.name, self.max_new_tokens = model_name, max_new_tokens

    def __call__(self, messages):
        import torch
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=self.max_new_tokens,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)  # greedy = temperature 0
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


class APIBackend:
    """Any OpenAI-compatible endpoint (e.g. Groq or Gemini free tiers).
    Check the provider's current free limits and terms before a full run."""

    def __init__(self, model_name, base_url, api_key, max_new_tokens=128, pause=0.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.name, self.max_new_tokens, self.pause = model_name, max_new_tokens, pause

    def __call__(self, messages):
        for attempt in range(5):
            try:
                r = self.client.chat.completions.create(model=self.name, messages=messages,
                                                        temperature=0, max_tokens=self.max_new_tokens)
                time.sleep(self.pause)
                return r.choices[0].message.content
            except Exception as e:  # rate limit / transient error: back off and retry
                wait = 2 ** attempt * 5
                print(f"API error ({e.__class__.__name__}), retry in {wait}s")
                time.sleep(wait)
        raise RuntimeError("API failed 5 times in a row")


# ---------- runs ----------

def load_done(path):
    """{(condition, question_id): record} already saved — lets a run resume after a Colab disconnect."""
    done = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                x = json.loads(line)
                done[(x["condition"], x["question_id"])] = x
    return done


def run_condition(condition, questions, context_fn, chunk_info, papers, llm, out_path, max_tokens=1280):
    """Generate answers for one condition, appending one json line per question.

    context_fn(record) -> ranked candidate chunk ids ([] for closed-book).
    """
    done = load_done(out_path)
    stale = [k for k, rec in done.items() if k[0] == condition and
             (rec.get("model") != llm.name or rec.get("prompt_version", PROMPT_VERSION) != PROMPT_VERSION)]
    if stale:
        raise SystemExit(f"{out_path} has {len(stale)} '{condition}' answers from another model or prompt "
                         f"version. Rerun with --fresh (or use another output file) instead of mixing runs.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as f:
        for i, r in enumerate(questions):
            if (condition, r["question_id"]) in done:
                continue
            ctx, used = build_context(context_fn(r), chunk_info, max_tokens)
            messages, labels = make_prompt(r["question"], papers[r["paper_id"]]["title"], ctx, chunk_info)
            t0 = time.time()
            raw = llm(messages)
            rec = {"condition": condition, "question_id": r["question_id"], "model": llm.name,
                   "prompt_version": PROMPT_VERSION,
                   "context_chunk_ids": ctx, "context_tokens": used, "raw": raw,
                   "latency_s": round(time.time() - t0, 2), **parse_output(raw, labels)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {condition}: {i + 1}/{len(questions)}")
    return load_done(out_path)
