from pathlib import Path
import re
import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

def load_scifact(data_dir="datasets", split="test"):
    dataset = "scifact"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, data_dir)
    return GenericDataLoader(data_folder=data_path).load(split=split)

def prepare_documents(corpus):
    doc_ids = list(corpus.keys())
    documents = [
        (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
        for d in doc_ids
    ]
    return doc_ids, documents

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def build_bm25(documents):
    return BM25Okapi([tokenize(doc) for doc in documents])

def retrieve_bm25(query, bm25, doc_ids, k=10):
    scores = bm25.get_scores(tokenize(query))
    idx = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in idx]

def load_dense_model(name="intfloat/multilingual-e5-base"):
    return SentenceTransformer(name)

def build_or_load_document_embeddings(
    dense_model, documents, doc_ids, cache_dir="cache", batch_size=32
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "scifact_multilingual_e5_base_embeddings.npy"
    ids_path = cache_dir / "scifact_doc_ids.npy"

    if emb_path.exists() and ids_path.exists():
        embeddings = np.load(emb_path)
        cached_ids = np.load(ids_path).tolist()
        if cached_ids != list(doc_ids):
            raise ValueError("Cached doc_ids do not match current corpus ordering.")
        return embeddings, cached_ids

    passages = [f"passage: {doc}" for doc in documents]
    embeddings = dense_model.encode(
        passages,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(emb_path, embeddings)
    np.save(ids_path, np.array(doc_ids))
    return embeddings, doc_ids

def retrieve_dense(query, dense_model, document_embeddings, doc_ids, k=10):
    q = dense_model.encode(f"query: {query}", normalize_embeddings=True)
    scores = document_embeddings @ q
    idx = np.argsort(scores)[::-1][:k]
    return [doc_ids[i] for i in idx]

def reciprocal_rank_fusion(rankings, rrf_k=60):
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return [d for d, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

def retrieve_hybrid(
    query, bm25, dense_model, document_embeddings, doc_ids,
    k=10, candidate_k=100, rrf_k=60
):
    b = retrieve_bm25(query, bm25, doc_ids, candidate_k)
    d = retrieve_dense(query, dense_model, document_embeddings, doc_ids, candidate_k)
    return reciprocal_rank_fusion([b, d], rrf_k)[:k]

def load_reranker(name="BAAI/bge-reranker-v2-m3"):
    return CrossEncoder(name)

def retrieve_reranked(
    query, corpus, bm25, dense_model, document_embeddings, doc_ids, reranker,
    k=10, candidate_k=20, hybrid_candidate_k=100
):
    candidates = retrieve_hybrid(
        query, bm25, dense_model, document_embeddings, doc_ids,
        k=candidate_k, candidate_k=hybrid_candidate_k
    )
    pairs = [
        (query, (corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip())
        for d in candidates
    ]
    scores = reranker.predict(pairs)
    idx = np.argsort(scores)[::-1][:k]
    return [candidates[i] for i in idx]
