import numpy as np


def recall_at_k(retrieved_doc_ids, gold_doc_ids):
    """Compute Recall@k for one query."""
    retrieved = set(retrieved_doc_ids)
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    return len(retrieved & gold) / len(gold)


def reciprocal_rank(retrieved_doc_ids, gold_doc_ids):
    """Compute reciprocal rank for one query."""
    gold = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def found_at_k(retrieved_doc_ids, gold_doc_ids, k):
    """Return whether at least one gold document appears in the top-k."""
    return bool(set(retrieved_doc_ids[:k]) & set(gold_doc_ids))


def evaluate_retriever(retrieve_fn, queries, qrels, ks=(1, 5, 10)):
    """Evaluate one ranking per query and derive all metrics from it."""
    max_k = max(ks)
    recall_values = {k: [] for k in ks}
    rr_values = {k: [] for k in ks}

    for query_id in qrels:
        query = queries[query_id]
        gold_doc_ids = set(qrels[query_id].keys())
        retrieved = retrieve_fn(query, max_k)

        for k in ks:
            top_k = retrieved[:k]
            recall_values[k].append(recall_at_k(top_k, gold_doc_ids))
            rr_values[k].append(reciprocal_rank(top_k, gold_doc_ids))

    results = {}
    for k in ks:
        results[f"Recall@{k}"] = float(np.mean(recall_values[k]))
        results[f"MRR@{k}"] = float(np.mean(rr_values[k]))
    return results


def find_bm25_fail_dense_success(queries, qrels, retrieve_bm25_fn, retrieve_dense_fn, k=10):
    """Find queries where BM25 fails and Dense succeeds."""
    cases = []
    for query_id in qrels:
        query = queries[query_id]
        gold_doc_ids = set(qrels[query_id].keys())
        bm25_results = retrieve_bm25_fn(query, k)
        dense_results = retrieve_dense_fn(query, k)
        if not found_at_k(bm25_results, gold_doc_ids, k) and found_at_k(dense_results, gold_doc_ids, k):
            cases.append(query_id)
    return cases


def find_dense_fail_bm25_success(queries, qrels, retrieve_bm25_fn, retrieve_dense_fn, k=10):
    """Find queries where Dense fails and BM25 succeeds."""
    cases = []
    for query_id in qrels:
        query = queries[query_id]
        gold_doc_ids = set(qrels[query_id].keys())
        bm25_results = retrieve_bm25_fn(query, k)
        dense_results = retrieve_dense_fn(query, k)
        if not found_at_k(dense_results, gold_doc_ids, k) and found_at_k(bm25_results, gold_doc_ids, k):
            cases.append(query_id)
    return cases


def find_dense_fail_hybrid_success(queries, qrels, retrieve_dense_fn, retrieve_hybrid_fn, k=10):
    """Find queries where Dense fails and Hybrid succeeds."""
    cases = []
    for query_id in qrels:
        query = queries[query_id]
        gold_doc_ids = set(qrels[query_id].keys())
        dense_results = retrieve_dense_fn(query, k)
        hybrid_results = retrieve_hybrid_fn(query, k)
        if not found_at_k(dense_results, gold_doc_ids, k) and found_at_k(hybrid_results, gold_doc_ids, k):
            cases.append(query_id)
    return cases


def find_reranker_improvements_batched(
    queries,
    qrels,
    corpus,
    retrieve_hybrid_fn,
    reranker,
    candidate_k=20,
    batch_size=64,
):
    """Find gold documents moved upward by reranking using batched inference."""
    all_pairs = []
    metadata = []

    for query_id in qrels:
        query = queries[query_id]
        hybrid_results = retrieve_hybrid_fn(query, candidate_k)

        for hybrid_rank, doc_id in enumerate(hybrid_results, start=1):
            document = (
                corpus[doc_id].get("title", "")
                + " "
                + corpus[doc_id].get("text", "")
            ).strip()
            all_pairs.append((query, document))
            metadata.append((query_id, doc_id, hybrid_rank))

    scores = reranker.predict(
        all_pairs,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    scores_by_query = {}
    for (query_id, doc_id, hybrid_rank), score in zip(metadata, scores):
        scores_by_query.setdefault(query_id, []).append(
            (doc_id, hybrid_rank, float(score))
        )

    cases = []
    for query_id, docs in scores_by_query.items():
        reranked = sorted(docs, key=lambda item: item[2], reverse=True)
        reranked_ids = [doc_id for doc_id, _, _ in reranked]
        gold_doc_ids = set(qrels[query_id].keys())

        for doc_id, hybrid_rank, _ in docs:
            if doc_id in gold_doc_ids:
                reranked_rank = reranked_ids.index(doc_id) + 1
                if reranked_rank < hybrid_rank:
                    cases.append((query_id, doc_id, hybrid_rank, reranked_rank))

    return cases
