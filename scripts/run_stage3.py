"""Stage 3 — grounded answer generation on QASPER (RQ3), as a script for a GPU server.

Three phases, run in this order:

  1. Prompt check on the DEV split (never on the test subset):
         python scripts/run_stage3.py --phase dev
     Prints raw model outputs and dev metrics. If the prompt in src/generation.py
     is changed, rerun with --fresh.

  2. Generation on the 300 test questions, 6 conditions (long: run with nohup):
         nohup python scripts/run_stage3.py --phase test > results/stage3_log.txt 2>&1 &
     Resumable: finished answers are skipped after a crash or restart.

  3. Evaluation only (no GPU needed), also run automatically at the end of phase 2:
         python scripts/run_stage3.py --phase eval

Needs results/qasper_rankings.jsonl from stage 2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import prepare_qasper
from src.generation import run_condition, oracle_chunks, build_context, make_prompt, parse_output, load_done
from src.answer_eval import evaluate_outputs, gold_answer_strings
from src.qasper_retrieval import paired_bootstrap

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
RESULTS = ROOT / "results"
TEST_OUT = RESULTS / "generation_test.jsonl"
DEV_OUT = RESULTS / "generation_dev.jsonl"
ORDER = ["closed-book", "BM25", "Dense", "Hybrid", "Hybrid+CE", "oracle"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")


def load_llm(args):
    from src.generation import HFBackend
    log("loading generator", args.model, "(4-bit)" if args.load_in_4bit else "(bfloat16)")
    return HFBackend(args.model, load_in_4bit=args.load_in_4bit)


# ---------------- phase 1: dev prompt check ----------------

def phase_dev(args):
    if args.fresh and DEV_OUT.exists():
        DEV_OUT.unlink()
    tok = load_tokenizer()
    dev = prepare_qasper(ROOT, tok, args.max_tokens, split="dev", n=args.n_dev,
                         subset_file="qasper_dev_prompt_subset.json")
    ci, ch = dev["chunk_info"][args.chunking], dev["chunks"][args.chunking]
    llm = load_llm(args)

    log("5 raw examples (oracle context):")
    for r in dev["eval_qs"][:5]:
        ctx, _ = build_context(oracle_chunks(r, ch[r["paper_id"]]), ci, args.budget)
        messages, labels = make_prompt(r["question"], dev["papers"][r["paper_id"]]["title"], ctx, ci)
        raw = llm(messages)
        print("Q:     ", r["question"])
        print("GOLD:  ", gold_answer_strings(r)[:2])
        print("RAW:   ", raw)
        print("PARSED:", parse_output(raw, labels), "\n", flush=True)

    out = run_condition("dev_oracle", dev["eval_qs"], lambda r: oracle_chunks(r, ch[r["paper_id"]]),
                        ci, dev["papers"], llm, DEV_OUT, args.budget)
    out.update(run_condition("dev_closed-book", dev["eval_qs"], lambda r: [],
                             ci, dev["papers"], llm, DEV_OUT, args.budget))
    ev = pd.DataFrame(evaluate_outputs(out, dev["eval_qs"], ci, dev["paragraphs"]))
    print(ev.groupby("condition").agg(answer_F1=("answer_f1", "mean"), valid_json=("parsed_json", "mean"),
                                      said_unanswerable=("said_unanswerable", "mean")).round(3).to_string(), flush=True)
    print(ev.pivot_table(index="condition", columns="answer_type", values="answer_f1", aggfunc="mean")
          .round(3).to_string(), flush=True)
    log("dev check done")


# ---------------- phase 2: test generation ----------------

def phase_test(args):
    if args.fresh and TEST_OUT.exists():
        TEST_OUT.unlink()
    tok = load_tokenizer()
    data = prepare_qasper(ROOT, tok, args.max_tokens)
    papers, eval_qs = data["papers"], data["eval_qs"]
    ci, ch = data["chunk_info"][args.chunking], data["chunks"][args.chunking]

    rankings = {}
    with open(RESULTS / "qasper_rankings.jsonl", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            if x["chunking"] == args.chunking:
                rankings.setdefault(x["method"], {})[x["question_id"]] = x["ranked"]
    missing = [cid for m in rankings.values() for ids in m.values() for cid in ids if cid not in ci]
    if missing:
        raise SystemExit(f"{len(missing)} chunk ids in qasper_rankings.jsonl are unknown: rerun stage 2 first.")

    conditions = {
        "closed-book": lambda r: [],
        "BM25": lambda r: rankings["BM25"][r["question_id"]],
        "Dense": lambda r: rankings["Dense"][r["question_id"]],
        "Hybrid": lambda r: rankings["Hybrid"][r["question_id"]],
        "Hybrid+CE": lambda r: rankings["Hybrid+CE"][r["question_id"]],
        "oracle": lambda r: oracle_chunks(r, ch[r["paper_id"]]),
    }
    llm = load_llm(args)
    for name in ORDER:
        log("condition", name)
        run_condition(name, eval_qs, conditions[name], ci, papers, llm, TEST_OUT, args.budget)
    log("generation done")
    phase_eval(args, data)


# ---------------- phase 3: evaluation ----------------

def phase_eval(args, data=None):
    if data is None:
        data = prepare_qasper(ROOT, load_tokenizer(), args.max_tokens)
    eval_qs, ci = data["eval_qs"], data["chunk_info"][args.chunking]
    outputs = load_done(TEST_OUT)
    ev = pd.DataFrame(evaluate_outputs(outputs, eval_qs, ci, data["paragraphs"]))
    ev.to_csv(RESULTS / "generation_per_question.csv", index=False)
    counts = ev.condition.value_counts()
    log("answers per condition:", counts.to_dict())

    order = [c for c in ORDER if c in counts]
    summary = ev.groupby("condition").agg(
        answer_F1=("answer_f1", "mean"),
        evidence_F1=("evidence_f1", "mean"),
        evidence_in_context=("evidence_in_context", "mean"),
        citation_rate=("has_citation", "mean"),
        citation_precision=("citation_precision", "mean"),  # among answers with >= 1 citation
        said_unanswerable=("said_unanswerable", "mean"),
        valid_json=("parsed_json", "mean"),
        context_tokens=("context_tokens", "mean"),
        latency_s=("latency_s", "mean"),
    ).reindex(order).round(3)
    if {"closed-book", "oracle"} <= set(order):
        lo, hi = summary.loc["closed-book", "answer_F1"], summary.loc["oracle", "answer_F1"]
        summary["gap_closed"] = ((summary["answer_F1"] - lo) / (hi - lo)).round(3) if hi != lo else None
    summary.to_csv(RESULTS / "generation_summary.csv")
    print("\n== Summary\n" + summary.to_string(), flush=True)

    by_type = ev.pivot_table(index="condition", columns="answer_type", values="answer_f1", aggfunc="mean").reindex(order).round(3)
    by_type.to_csv(RESULTS / "generation_f1_by_type.csv")
    n_type = ev[ev.condition == order[0]].answer_type.value_counts().to_dict()
    print("\n== Answer-F1 by type of the best-matching annotator (official grouping)", n_type)
    print(by_type.to_string(), flush=True)

    retr = [c for c in ["BM25", "Dense", "Hybrid", "Hybrid+CE"] if c in order]
    rq3 = (ev[ev.condition.isin(retr)]
           .groupby(["condition", "evidence_in_context"]).answer_f1.agg(["mean", "count"])
           .unstack("evidence_in_context").reindex(retr).round(3))
    rq3.to_csv(RESULTS / "generation_f1_by_evidence.csv")
    print("\n== RQ3: Answer-F1 with / without gold evidence in context\n" + rq3.to_string(), flush=True)

    def f1_vec(c):
        s = ev[ev.condition == c].set_index("question_id")["answer_f1"]
        return [s[r["question_id"]] for r in eval_qs]

    pairs = [("closed-book", "Hybrid+CE"), ("BM25", "Dense"), ("Dense", "Hybrid"), ("Hybrid", "Hybrid+CE"),
             ("Dense", "Hybrid+CE"), ("Hybrid+CE", "oracle"), ("Dense", "oracle")]
    rows = []
    for a, b in pairs:
        if counts.get(a, 0) == len(eval_qs) and counts.get(b, 0) == len(eval_qs):
            d, l, h = paired_bootstrap(f1_vec(a), f1_vec(b))
            rows.append({"A": a, "B": b, "delta Answer-F1": round(d, 3), "95% CI": f"[{l:.3f}, {h:.3f}]",
                         "significant": not (l <= 0 <= h)})
    boot = pd.DataFrame(rows)
    boot.to_csv(RESULTS / "generation_bootstrap.csv", index=False)
    print("\n== Paired bootstrap (Answer-F1)\n" + boot.to_string(index=False), flush=True)

    # error analysis for Hybrid+CE
    qs = {r["question_id"]: r for r in eval_qs}
    lines = []
    if "Hybrid+CE" in order:
        ce = ev[ev.condition == "Hybrid+CE"]
        for title, sub in [("RETRIEVAL FAILURE (no gold evidence in context)", ce[~ce.evidence_in_context]),
                           ("GENERATION FAILURE (gold evidence in context, F1 < 0.3)",
                            ce[ce.evidence_in_context & (ce.answer_f1 < 0.3)])]:
            lines.append(f"===== {title}: {len(sub)} questions =====\n")
            for qid in sub.question_id[:8]:
                o = outputs[("Hybrid+CE", qid)]
                orc = outputs.get(("oracle", qid), {}).get("answer")
                lines += [f"Q: {qs[qid]['question']}  [{qid}]",
                          f"GOLD: {gold_answer_strings(qs[qid])}",
                          f"Hybrid+CE: {o['answer']}", f"oracle: {orc}", ""]
    (RESULTS / "generation_error_analysis.txt").write_text("\n".join(lines), encoding="utf-8")

    # predictions in the official evaluator format, for cross-checking Answer-F1
    paras = data["paragraphs"]
    for cond in order:
        with open(RESULTS / f"predictions_{cond}.jsonl", "w", encoding="utf-8") as f:
            for (c, qid), o in outputs.items():
                if c != cond:
                    continue
                pid = qs[qid]["paper_id"]
                ev_paras = sorted({p for cid in o["cited_chunk_ids"] for p in ci[cid]["para_ids"]})
                f.write(json.dumps({"question_id": qid, "predicted_answer": o["answer"],
                                    "predicted_evidence": [paras[pid][p]["text"] for p in ev_paras]}) + "\n")
    official_check(data, order)
    log("evaluation done; files in", RESULTS)


def official_check(data, order):
    """Run the official QASPER evaluator on the same 300 questions, if it is in the test archive."""
    import subprocess
    found = list((ROOT / "datasets" / "qasper").rglob("qasper_evaluator.py"))
    if not found:
        log("official evaluator not found in datasets/qasper; skipping the cross-check")
        return
    subset = {r["question_id"] for r in data["eval_qs"]}
    gold = {}
    for pid, p in data["papers"].items():
        qas = [q for q in p["qas"] if q["question_id"] in subset]
        if qas:
            gold[pid] = {**p, "qas": qas}
    gold_path = ROOT / "datasets" / "qasper" / "qasper_test_subset_gold.json"
    gold_path.write_text(json.dumps(gold), encoding="utf-8")
    for cond in order:
        pred = RESULTS / f"predictions_{cond}.jsonl"
        for extra in (["--text_evidence_only"], []):
            res = subprocess.run([sys.executable, str(found[0]), "--predictions", str(pred), "--gold", str(gold_path), *extra],
                                 capture_output=True, text=True)
            if res.returncode == 0:
                break
        out = res.stdout if res.returncode == 0 else res.stdout + res.stderr
        (RESULTS / f"official_eval_{cond}.txt").write_text(out, encoding="utf-8")
        print(f"\n== official evaluator, {cond}{' (text evidence only)' if extra else ''}\n{out.strip()[-600:]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "test", "eval"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--chunking", default="paragraph+section")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--budget", type=int, default=1280, help="max context tokens per question")
    ap.add_argument("--n-dev", type=int, default=30)
    ap.add_argument("--fresh", action="store_true", help="delete previous outputs of this phase (dev or test) first")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    {"dev": phase_dev, "test": phase_test, "eval": phase_eval}[args.phase](args)
