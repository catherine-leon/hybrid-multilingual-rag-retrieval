"""Per-paper retrieval on QASPER, reusing the SciFact functions from retrieval.py.

In QASPER each question is asked about ONE paper, so the search space is
the chunks of that paper (tens of chunks), not the whole collection.
"""
from pathlib import Path

import numpy as np

from src.retrieval import (
    build_bm25,
    retrieve_bm25,
    retrieve_dense,
    reciprocal_rank_fusion,
)
from src.evaluation import recall_at_k, reciprocal_rank


class PaperIndex:
    """BM25 + dense index over the chunks of a single paper."""

    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.text = {c["chunk_id"]: c["text"] for c in chunks}
        self.bm25 = build_bm25([c["text"] for c in chunks])
        self.embeddings = embeddings  # rows aligned with self.chunk_ids

    def bm25_search(self, query, k):
        return retrieve_bm25(query, self.bm25, self.chunk_ids, k)

    def dense_search(self, query, dense_model, k):
        return retrieve_dense(query, dense_model, self.embeddings, self.chunk_ids, k)

    def hybrid_search(self, query, dense_model, k, candidate_k=100, rrf_k=60):
        b = self.bm25_search(query, candidate_k)
        d = self.dense_search(query, dense_model, candidate_k)
        return reciprocal_rank_fusion([b, d], rrf_k)[:k]

    def reranked_search(self, query, dense_model, reranker, k, candidate_k=20):
        candidates = self.hybrid_search(query, dense_model, k=candidate_k)
        scores = reranker.predict([(query, self.text[c]) for c in candidates])
        order = np.argsort(scores)[::-1][:k]
        return [candidates[i] for i in order]


def embed_all_chunks(dense_model, chunks_by_paper, cache_path=None, batch_size=64):
    """Encode all chunks of all papers in one pass (fast on GPU), with caching.

    Returns {paper_id: embedding matrix aligned with that paper's chunks}.
    """
    all_chunks = [c for pid in chunks_by_paper for c in chunks_by_paper[pid]]
    all_ids = [c["chunk_id"] for c in all_chunks]

    if cache_path is not None and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=False)
        if data["ids"].tolist() != all_ids:
            raise ValueError("Cached chunk ids do not match current chunks. Delete the cache.")
        emb = data["emb"]
    else:
        emb = dense_model.encode(
            [f"passage: {c['text']}" for c in all_chunks],
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, emb=emb, ids=np.array(all_ids))

    out, start = {}, 0
    for pid, chunks in chunks_by_paper.items():
        out[pid] = emb[start:start + len(chunks)]
        start += len(chunks)
    return out


def evaluate_qasper(search_fn, questions, gold_by_question, chunk_info, ks=(1, 3, 5)):
    """search_fn(record, k) -> ranked chunk ids for that record's paper.

    Same logic as evaluate_retriever(): one ranking per question at max k,
    all metrics derived from it.

    chunk_info: {chunk_id: chunk dict} — used for two chunking-fair metrics:
      ParaRecall@k — share of gold PARAGRAPHS covered by the top-k chunks
                     (comparable across chunking strategies);
      Tokens@k     — size of the top-k chunks in retriever tokens (context
                     cost: bigger chunks find evidence more easily but cost more).

    Returns (metrics, rankings, per_question):
      rankings     = {question_id: ranked chunk ids}
      per_question = {metric name: list of per-question values, in `questions`
                      order} — used for paired bootstrap.
    """
    max_k = max(ks)
    per_q = {f"{m}@{k}": [] for m in ("Recall", "MRR", "ParaRecall", "Tokens") for k in ks}
    rankings = {}

    for r in questions:
        gold = set(gold_by_question[r["question_id"]])
        gold_paras = set(r["gold_paragraphs"])
        ranked = search_fn(r, max_k)
        rankings[r["question_id"]] = ranked
        for k in ks:
            top = ranked[:k]
            covered = set().union(*(chunk_info[c]["para_ids"] for c in top)) if top else set()
            per_q[f"Recall@{k}"].append(recall_at_k(top, gold))
            per_q[f"MRR@{k}"].append(reciprocal_rank(top, gold))
            per_q[f"ParaRecall@{k}"].append(len(covered & gold_paras) / len(gold_paras))
            per_q[f"Tokens@{k}"].append(sum(chunk_info[c]["n_tokens"] for c in top))

    metrics = {name: float(np.mean(v)) for name, v in per_q.items()}
    return metrics, rankings, per_q


def paired_bootstrap(a, b, n_boot=1000, seed=0):
    """Paired bootstrap over questions for mean(b) - mean(a).

    Returns (difference, 95% CI low, 95% CI high). If the interval contains 0,
    the difference is within noise for this sample size.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (b[idx] - a[idx]).mean(axis=1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(b.mean() - a.mean()), float(lo), float(hi)
