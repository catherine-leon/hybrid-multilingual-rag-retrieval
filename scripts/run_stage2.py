"""Stage 2 — QASPER chunking + in-paper retrieval, as a script for a GPU server.

Same steps as notebooks/02_qasper_chunking_retrieval.ipynb.

Run from the repository root:
    python scripts/run_stage2.py
Long run, keep it alive after logout:
    nohup python scripts/run_stage2.py > results/stage2_log.txt 2>&1 &

Outputs in results/:
    qasper_eval_subset.json, qasper_retrieval_metrics.csv, qasper_bootstrap.csv,
    qasper_rankings.jsonl, qasper_chunking_stats.csv, qasper_error_analysis.txt
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.retrieval import load_dense_model
from src.qasper import download_qasper, load_qasper_split, paper_paragraphs, extract_questions, sample_eval_subset
from src.chunking import make_chunkers, gold_chunk_ids
from src.qasper_retrieval import PaperIndex, embed_all_chunks, evaluate_qasper, paired_bootstrap

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def main(args):
    results = ROOT / "results"
    results.mkdir(exist_ok=True)

    # ---- 1-2. data + gold evidence
    data_dir = ROOT / "datasets" / "qasper"
    if not args.skip_download:
        download_qasper(data_dir)
    papers = load_qasper_split("test", data_dir)
    records, ev_stats = extract_questions(papers)
    log("papers:", len(papers), "| questions:", len(records))
    log("evidence stats:", ev_stats)
    log("questions with text evidence:", sum(bool(r["gold_paragraphs"]) for r in records), "/", len(records))

    # ---- 3. fixed evaluation subset
    eval_qs = sample_eval_subset(records, n=args.n, seed=42, path=results / "qasper_eval_subset.json")
    eval_papers = sorted({r["paper_id"] for r in eval_qs})
    paragraphs_by_paper = {pid: paper_paragraphs(papers[pid]) for pid in eval_papers}
    log(len(eval_qs), "questions from", len(eval_papers), "papers")

    # ---- 4. token-based chunking
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.dense_model)
    M = args.max_tokens
    chunks = {name: {pid: ch(pid, paragraphs_by_paper[pid]) for pid in eval_papers}
              for name, ch in make_chunkers(tokenizer, M).items()}

    ntok = lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"])
    para_tokens = {(pid, p["para_idx"]): ntok(p["text"]) for pid, ps in paragraphs_by_paper.items() for p in ps}
    n_long = sum(t > M for t in para_tokens.values())
    gold_pairs = [(r["paper_id"], i) for r in eval_qs for i in r["gold_paragraphs"]]
    long_gold = sum(para_tokens[g] > M for g in gold_pairs)
    log(f"paragraphs longer than {M} tokens: {n_long} / {len(para_tokens)} ({100*n_long/len(para_tokens):.1f}%)")
    log(f"longest paragraph: {max(para_tokens.values())} tokens")
    log(f"gold paragraphs split into several chunks: {long_gold} / {len(gold_pairs)}")

    rows = []
    for name, by_paper in chunks.items():
        toks = [c["n_tokens"] for cs in by_paper.values() for c in cs]
        rows.append({"chunking": name, "chunks/paper": round(len(toks) / len(by_paper), 1),
                     "mean tokens": round(sum(toks) / len(toks), 1), "max tokens": max(toks)})
    stats = pd.DataFrame(rows)
    stats.to_csv(results / "qasper_chunking_stats.csv", index=False)
    print(stats.to_string(index=False), flush=True)

    # ---- 5. models + indexes
    from sentence_transformers import CrossEncoder
    dense_model = load_dense_model(args.dense_model)
    dense_model.max_seq_length = 512
    reranker = CrossEncoder(args.reranker, max_length=512)

    indexes, chunk_info, gold = {}, {}, {}
    for name, by_paper in chunks.items():
        log("embedding", name)
        emb = embed_all_chunks(dense_model, by_paper, cache_path=ROOT / "cache" / f"qasper_e5_{name}_{M}tok.npz",
                               batch_size=args.batch_size)
        indexes[name] = {pid: PaperIndex(by_paper[pid], emb[pid]) for pid in eval_papers}
        chunk_info[name] = {c["chunk_id"]: c for cs in by_paper.values() for c in cs}
        gold[name] = {r["question_id"]: gold_chunk_ids(by_paper[r["paper_id"]], r["gold_paragraphs"]) for r in eval_qs}

    # ---- 6. evaluation
    def search_fns(name):
        idx = indexes[name]
        return {
            "BM25": lambda r, k: idx[r["paper_id"]].bm25_search(r["question"], k),
            "Dense": lambda r, k: idx[r["paper_id"]].dense_search(r["question"], dense_model, k),
            "Hybrid": lambda r, k: idx[r["paper_id"]].hybrid_search(r["question"], dense_model, k),
            "Hybrid+CE": lambda r, k: idx[r["paper_id"]].reranked_search(r["question"], dense_model, reranker, k),
        }

    res, all_rankings, per_question = [], {}, {}
    for name in chunks:
        for method, fn in search_fns(name).items():
            m, rankings, per_q = evaluate_qasper(fn, eval_qs, gold[name], chunk_info[name])
            res.append({"chunking": name, "method": method, **m})
            all_rankings[(name, method)] = rankings
            per_question[(name, method)] = per_q
            log(name, method, "done")
    df = pd.DataFrame(res).round(3)
    df.to_csv(results / "qasper_retrieval_metrics.csv", index=False)
    print(df[["chunking", "method", "ParaRecall@1", "ParaRecall@3", "ParaRecall@5", "MRR@5", "Tokens@5"]]
          .to_string(index=False), flush=True)

    # ---- 7. paired bootstrap
    P, PS, F = "paragraph", "paragraph+section", f"fixed{M}"
    comparisons = [((P, "BM25"), (P, "Dense")), ((P, "Dense"), (P, "Hybrid")),
                   ((P, "Hybrid"), (P, "Hybrid+CE")), ((P, "Dense"), (P, "Hybrid+CE")),
                   ((P, "BM25"), (PS, "BM25")), ((P, "Hybrid+CE"), (PS, "Hybrid+CE")),
                   ((PS, "Hybrid+CE"), (F, "Hybrid+CE"))]
    metric = "ParaRecall@5"
    brows = []
    for a, b in comparisons:
        d, lo, hi = paired_bootstrap(per_question[a][metric], per_question[b][metric])
        brows.append({"A": " / ".join(a), "B": " / ".join(b), f"delta {metric}": round(d, 3),
                      "95% CI": f"[{lo:.3f}, {hi:.3f}]", "significant": not (lo <= 0 <= hi)})
    boot = pd.DataFrame(brows)
    boot.to_csv(results / "qasper_bootstrap.csv", index=False)
    print(boot.to_string(index=False), flush=True)

    # ---- 8. rankings for stage 3
    with open(results / "qasper_rankings.jsonl", "w", encoding="utf-8") as f:
        for (name, method), rankings in all_rankings.items():
            for qid, ranked in rankings.items():
                f.write(json.dumps({"chunking": name, "method": method, "question_id": qid, "ranked": ranked}) + "\n")

    # ---- 9. error analysis -> text file
    gold_sets = {name: {q: set(v) for q, v in gold[name].items()} for name in chunks}
    qs = {r["question_id"]: r for r in eval_qs}

    def gold_rank(name, method, qid, k=5):
        for i, c in enumerate(all_rankings[(name, method)][qid][:k], 1):
            if c in gold_sets[name][qid]:
                return i
        return None

    cases = {"BM25 fails, Dense succeeds": ((P, "BM25"), (P, "Dense")),
             "Dense succeeds, Hybrid fails (RRF hurts)": ((P, "Hybrid"), (P, "Dense")),
             "Hybrid fails, Hybrid+CE succeeds": ((P, "Hybrid"), (P, "Hybrid+CE")),
             "Section name helps BM25": ((P, "BM25"), (PS, "BM25"))}
    lines = []
    for title, (fail, ok) in cases.items():
        ids = [q for q in qs if gold_rank(*fail, q) is None and gold_rank(*ok, q) is not None]
        lines.append(f"===== {title}: {len(ids)} questions =====\n")
        for qid in ids[:5]:
            g = sorted(gold_sets[fail[0]][qid])[0]
            lines.append(f"QUESTION: {qs[qid]['question']}  [{qid}]")
            lines.append(f"GOLD: {chunk_info[fail[0]][g]['text'][:400]} ...")
            for name, m in (fail, ok):
                top1 = chunk_info[name][all_rankings[(name, m)][qid][0]]["text"][:200]
                lines.append(f"  {name} / {m}: gold at rank {gold_rank(name, m, qid)} | top-1: {top1}...")
            lines.append("")
    (results / "qasper_error_analysis.txt").write_text("\n".join(lines), encoding="utf-8")
    log("done. results in", results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="evaluation subset size")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dense-model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--reranker", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--skip-download", action="store_true", help="use already extracted data in datasets/qasper")
    main(ap.parse_args())
