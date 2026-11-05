"""Stage 3: answer and citation evaluation.

Answer-F1 follows the logic of the official QASPER evaluator: normalised
token F1 against each annotator's answer, maximum over annotators.
Gold answer strings: extractive -> spans joined with ", ", abstractive ->
free-form answer, yes/no -> "Yes"/"No", unanswerable -> "Unanswerable".
Cross-check on a few questions with qasper_evaluator.py from the official
test archive before reporting numbers.
"""
import re
import string
from collections import Counter


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(prediction, gold):
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p and not g:
        return 1.0
    common = Counter(p) & Counter(g)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision, recall = n_same / len(p), n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def gold_answer_strings(record):
    out = []
    for a in record["answers"]:
        if a["type"] == "extractive":
            out.append(", ".join(a["extractive_spans"]))
        elif a["type"] == "abstractive":
            out.append(a["free_form_answer"])
        elif a["type"] == "boolean":
            out.append("Yes" if a["yes_no"] else "No")
        else:
            out.append("Unanswerable")
    return out


def answer_f1(prediction, record):
    """Max token F1 over annotators; also returns the type of the best-matching gold answer."""
    scores = [(token_f1(prediction, g), a["type"])
              for g, a in zip(gold_answer_strings(record), record["answers"])]
    return max(scores, key=lambda x: x[0])


def evidence_f1(predicted_texts, record):
    """Official-style Evidence-F1: set F1 between predicted evidence paragraphs and
    each annotator's evidence paragraphs, maximum over annotators.
    Both empty -> 1.0; only one empty -> 0.0. Text evidence only (no figures/tables)."""
    pred = set(predicted_texts)
    best = 0.0
    for a in record["answers"]:
        gold = set(a.get("evidence", []))
        if not pred and not gold:
            f = 1.0
        elif not pred or not gold:
            f = 0.0
        else:
            tp = len(pred & gold)
            f = 0.0 if tp == 0 else 2 * tp / (len(pred) + len(gold))
        best = max(best, f)
    return best


def citation_scores(cited_chunk_ids, context_chunk_ids, record, chunk_info):
    """Citation precision: share of cited chunks that contain gold evidence.
    evidence_in_context: whether ANY gold paragraph was in the context at all
    (separates retrieval failures from generation failures)."""
    gold = set(record["gold_paragraphs"])
    has_gold = lambda cid: bool(gold & set(chunk_info[cid]["para_ids"]))
    cited_ok = [has_gold(c) for c in cited_chunk_ids]
    return {
        "citation_precision": sum(cited_ok) / len(cited_ok) if cited_ok else None,
        "n_cited": len(cited_chunk_ids),
        "evidence_in_context": any(has_gold(c) for c in context_chunk_ids),
    }


def evaluate_outputs(outputs, questions, chunk_info, paragraphs=None):
    """outputs: {(condition, qid): record}. Returns a list of per-question rows.

    answer_type is the type of the best-matching annotator (as in the official
    evaluator); first_annotator_type is kept for reference.
    paragraphs ({paper_id: paragraph list}) enables Evidence-F1 from the cited chunks.
    """
    qs = {r["question_id"]: r for r in questions}
    rows = []
    for (condition, qid), o in outputs.items():
        r = qs[qid]
        f1, gold_type = answer_f1(o["answer"], r)
        ev_f1 = None
        if paragraphs is not None:
            para_ids = sorted({p for c in o["cited_chunk_ids"] for p in chunk_info[c]["para_ids"]})
            ev_f1 = evidence_f1([paragraphs[r["paper_id"]][p]["text"] for p in para_ids], r)
        rows.append({
            "condition": condition, "question_id": qid,
            "answer_type": gold_type,
            "first_annotator_type": r["answers"][0]["type"],
            "answer_f1": f1,
            "evidence_f1": ev_f1,
            "has_citation": len(o["cited_chunk_ids"]) > 0,
            "said_unanswerable": normalize_answer(o["answer"]) == "unanswerable",
            "parsed_json": o["parsed_json"],
            "context_tokens": o["context_tokens"],
            "latency_s": o["latency_s"],
            **citation_scores(o["cited_chunk_ids"], o["context_chunk_ids"], r, chunk_info),
        })
    return rows
