# Hybrid and Cross-lingual Retrieval for Scientific RAG

## Motivation and research questions

A retrieval-augmented generation (RAG) system can only ground its answer in evidence that the retriever actually finds. Retrieval methods are usually compared on corpus-level IR benchmarks, where the task is to find relevant documents in a large collection. Question answering over a scientific paper poses a different problem. The evidence has to be located *inside* one long document, under a limited context budget, and the user's question may be in a different language than the paper.

This project studies how retrieval design choices behave in that setting:

| | Research question | Stage | Status |
|---|---|---|---|
| **RQ1** | Do conclusions about sparse, dense, hybrid and reranked retrieval drawn on a corpus-level benchmark transfer to evidence retrieval within scientific papers? | 1–2 | ✅ |
| **RQ2** | How does the chunking strategy affect evidence coverage at a fixed context budget? | 2 | ✅ |
| **RQ3** | Does better evidence retrieval translate into more accurate and more faithful generated answers? | 3 | planned |
| **RQ4** | How robust is the pipeline when the questions are asked in French or Russian about English papers? | 4 | planned |
| **RQ5** | Can adaptive retrieval (evidence check → query rewriting → retry) recover missing evidence, and at what cost? | 5 | planned |

## Retrieval methods

The same four retrievers are used in all stages:

```text
Query ──┬── BM25 ───────────┐
        └── Dense (mE5) ────┴── Reciprocal Rank Fusion ── top-20 ── Cross-Encoder ── final ranking
```

| Method | Implementation | Signal |
|---|---|---|
| BM25 | `rank-bm25` | lexical overlap |
| Dense | `intfloat/multilingual-e5-base`, `query:` / `passage:` prefixes, cosine similarity | semantic similarity, bi-encoder |
| Hybrid | Reciprocal Rank Fusion, `Σ 1 / (60 + rank)`, top 100 of each retriever | combination of both rankings |
| Hybrid + CE | `BAAI/bge-reranker-v2-m3` reranks the top 20 hybrid candidates | joint query–passage relevance |

The models are multilingual on purpose, so that the same pipeline can be evaluated on French and Russian questions (RQ4).

Metrics: **Recall@k** (share of gold evidence in the top k) and **MRR@k** (rank of the first relevant item). For each query, retrieval runs once at the maximum k, and all metrics are computed from that single ranking.

## Datasets: from corpus-level retrieval to evidence retrieval in full papers

The two datasets cover the two settings that RQ1 compares.

| | SciFact (BEIR) | QASPER (Dasigi et al., 2021) |
|---|---|---|
| Task | scientific claim verification | question answering over full NLP papers |
| Queries | scientific claims | information-seeking questions, written from the title and abstract only |
| Retrieval unit | abstract | paragraph / chunk |
| Search space | whole corpus | the chunks of a single paper |
| Gold annotation | expert relevance labels (relevant abstracts) | evidence paragraphs **and** answers |
| Supports evaluation of | retrieval | retrieval, chunking, generation |

SciFact provides a controlled, widely used reference point for the four retrievers. QASPER makes it possible to test whether the same conclusions hold when the evidence has to be localised inside a long document. It also exposes two factors that SciFact cannot measure: chunking (RQ2) and answer quality (RQ3). All further stages use QASPER.

## Stage 1: SciFact (corpus-level retrieval)

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@1 | MRR@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.508 | 0.728 | 0.774 | 0.523 | 0.613 | 0.619 |
| Dense | 0.557 | 0.760 | 0.807 | 0.583 | 0.660 | 0.666 |
| Hybrid RRF | 0.563 | 0.743 | 0.833 | 0.583 | 0.657 | 0.668 |
| **Hybrid + CE** | **0.613** | **0.787** | **0.836** | **0.643** | **0.710** | **0.715** |

- **Dense improves over the lexical baseline** on every reported metric, which shows the benefit of semantic matching when the query and the document use different vocabulary.
- **Hybrid improves candidate coverage.** It reaches a higher Recall@10 than either BM25 or Dense alone, which suggests that lexical and semantic retrieval recover partly complementary relevant documents. At k = 5, Hybrid stays below Dense (0.743 vs 0.760).
- **Reranking improves top-rank quality.** Compared with Hybrid, Recall@1 rises from 0.563 to 0.613, MRR@1 from 0.583 to 0.643 and MRR@5 from 0.657 to 0.710. Recall@10 changes only slightly, because reranking mostly reorders documents already present in the candidate set. Across SciFact, the cross-encoder moved a gold document to a higher position in 67 cases.

| Error pattern (top 10) | Query | Claim | Behaviour |
|---|---|---|---|
| BM25 fails, Dense succeeds | 239 | *Cellular aging closely links to an older appearance.* | BM25 retrieves literal "cellular aging" documents and misses the gold one. Dense ranks it 3rd, linking *older appearance* to biological ageing. |
| Dense fails, BM25 succeeds | 72 | *Activator-inhibitor pairs are provided dorsally by Admpchordin.* | BM25: rank 9. Dense misses it in the top 10. The gold paper explicitly discusses ADMP and Chordin, so exact domain terminology favours lexical retrieval. |
| Dense fails, Hybrid succeeds | 70 | *Activation of PPM1D suppresses p53 function.* | Dense returns general p53 documents and misses both gold ones. Hybrid retrieves both PPM1D-specific documents at ranks 2 and 6. |
| Reranking improves Hybrid | 185 | *Breast cancer development is determined exclusively by genetic factors.* | The gold study (*Gene-environment interactions in 7610 women with breast cancer*) moves from rank 6 to rank 1. |

Details: `results/error_analysis.md`.

## Stage 2: QASPER (evidence retrieval within papers)

### Data preparation

Each paper is flattened into paragraphs, and every annotated evidence string is mapped to its paragraph. Evidence from all annotators is merged.

| Evidence statistics (test split) | Count | Share |
|---|---:|---:|
| Evidence items | 5,744 | |
| — pointing to figures / tables (excluded) | 459 | 8.0% |
| — text evidence not matched to a paragraph | 222 | 4.2% of text evidence |
| Questions with text evidence | 1,309 / 1,451 | 90.2% |

| Answer type (first annotator) | Questions | Share |
|---|---:|---:|
| extractive | 732 | 50.4% |
| abstractive | 371 | 25.6% |
| yes / no | 209 | 14.4% |
| unanswerable | 139 | 9.6% |

The evaluation subset contains 300 questions with text evidence, sampled with a fixed seed (`results/qasper_eval_subset.json`). The same subset is reused in all later stages.

### Chunking strategies

| Strategy | Description |
|---|---|
| paragraph | one chunk = one native paragraph |
| paragraph+section | paragraph prefixed with its section name (e.g. *"Experiments ::: Datasets. …"*) |
| fixed200 | 200-word windows with a 50-word overlap, ignoring paragraph and section boundaries |

Chunk sizes differ between strategies, so Recall@k over chunks is not comparable across them. The strategies are therefore compared with:
- **ParaRecall@k**: the share of gold paragraphs covered by the top-k chunks;
- **Words@k**: the size of the context that would be passed to the LLM.

### Results

| Chunking | Method | ParaRecall@1 | ParaRecall@3 | ParaRecall@5 | MRR@5 | Words@5 |
|---|---|---:|---:|---:|---:|---:|
| paragraph | BM25 | 0.114 | 0.297 | 0.376 | 0.319 | 532 |
| paragraph | Dense | 0.196 | 0.399 | 0.522 | 0.481 | 502 |
| paragraph | Hybrid | 0.176 | 0.371 | 0.473 | 0.450 | 554 |
| paragraph | Hybrid+CE | 0.207 | 0.411 | 0.557 | 0.490 | 573 |
| paragraph+section | BM25 | 0.124 | 0.304 | 0.405 | 0.337 | 559 |
| paragraph+section | Dense | 0.198 | 0.394 | 0.535 | 0.484 | 516 |
| paragraph+section | Hybrid | 0.166 | 0.389 | 0.500 | 0.448 | 574 |
| paragraph+section | **Hybrid+CE** | **0.207** | **0.436** | **0.584** | **0.497** | 579 |
| fixed200 | BM25 | 0.190 | 0.435 | 0.594 | 0.436 | 986 |
| fixed200 | Dense | 0.293 | 0.555 | 0.672 | 0.558 | 984 |
| fixed200 | Hybrid | 0.269 | 0.529 | 0.688 | 0.529 | 987 |
| fixed200 | **Hybrid+CE** | **0.324** | **0.601** | **0.743** | **0.606** | 986 |

Full metrics: `results/qasper_retrieval_metrics.csv`.

### RQ1: do the SciFact conclusions transfer?

With paragraph chunks, one chunk equals one paragraph, so ParaRecall@5 on QASPER and Recall@5 on SciFact measure the same thing.

| Difference at k = 5 | SciFact (Recall@5) | QASPER, paragraph (ParaRecall@5) |
|---|---:|---:|
| Dense − BM25 | +0.032 | **+0.146** |
| Hybrid − Dense | −0.017 | **−0.049** |
| Hybrid+CE − Hybrid | +0.044 | **+0.084** |

**The direction of every effect transfers, but each effect is roughly 2–4.5 times larger on QASPER.**
- The gap between lexical and semantic retrieval widens. QASPER questions were written without access to the paper's full text, so they rarely reuse its wording.
- Rank fusion costs more: when BM25 is weak, RRF pulls Dense's good rankings down.
- Reranking is worth twice as much, and it recovers more than the loss caused by fusion.

### RQ2: chunking at a fixed context budget

At equal k, fixed200 looks clearly best, but it passes roughly twice as much text. The fair comparison puts the strategies at a similar context size of about 500–600 words:

| Method | paragraph+section @5 | words | fixed200 @3 | words |
|---|---:|---:|---:|---:|
| BM25 | 0.405 | 559 | 0.435 | 593 |
| Dense | 0.535 | 516 | 0.555 | 589 |
| Hybrid | 0.500 | 574 | 0.529 | 593 |
| Hybrid+CE | 0.584 | 579 | 0.601 | 593 |

- **At a comparable budget, fixed windows keep a small advantage (+0.017 to +0.030).** Most of their apparent lead at equal k comes from the larger context.
- **Adding the section name gives a small gain** (+0.03 ParaRecall@5 for BM25 and Hybrid+CE over plain paragraphs).
- **Even the best configuration covers only about 60% of the gold evidence within ~600 words.** This motivates adaptive retrieval (RQ5).

### Error analysis

| Error pattern (top 5, paragraph chunks) | Questions (of 300) |
|---|---:|
| BM25 fails, Dense succeeds | 73 |
| Dense succeeds, Hybrid fails | 31 |
| Hybrid fails, Hybrid+CE succeeds | 44 |
| Section name makes BM25 succeed | 11 |

| Pattern | Question | Gold evidence | Behaviour |
|---|---|---|---|
| Lexical mismatch | *What languages does the new dataset contain?* | *"the first trilingual dataset comprising English, French, and Arabic tweets"* | BM25: not in top 5; Dense: rank 1 |
| Fusion hurts, reranking repairs | *How many languages are in the dataset?* | *"76 documents in English and 47 in Spanish"* | Dense: 1; Hybrid: not in top 5; Hybrid+CE: 2 |
| Relevant but unannotated chunk | *What domain of text are they working with?* | another paragraph (Dense: rank 5) | Dense's top-1 chunk (*"we retrieved hundreds of thousands of news articles and their corresponding highlights from DailyMail"*) is not annotated as evidence |

In the last case, the top-ranked chunk names the data source the question asks about, yet it is counted as a miss. This suggests that the evidence annotation is not exhaustive, and the scores may underestimate retrieval quality.

## Next steps

| Stage | Research question | Plan |
|---|---|---|
| 3. Generation | RQ3 | An open LLM answers from the retrieved chunks and cites chunk IDs. Evaluation: official QASPER answer F1 by answer type, unanswerable detection, faithfulness on a subsample. Main comparison: the same LLM with each retriever. |
| 4. Cross-lingual | RQ4 | The 300 questions are machine-translated into French and Russian, and a sample is manually verified. Comparison: direct multilingual retrieval vs translating the question into English first, and the drop relative to English. |
| 5. Adaptive retrieval | RQ5 | Evidence-sufficiency check → query rewriting / translation → retry. Static vs adaptive: answer quality, evidence recall, number of retries, latency. |

## Repository structure

```text
├── notebooks/
│   ├── 01_scifact_retrieval_benchmark.ipynb
│   └── 02_qasper_chunking_retrieval.ipynb
├── src/
│   ├── retrieval.py          # BM25, dense, RRF, reranking (shared)
│   ├── evaluation.py         # Recall@k, MRR@k, error-case search
│   ├── qasper.py             # QASPER loading, evidence → paragraph mapping, eval subset
│   ├── chunking.py           # chunking strategies + gold chunk mapping
│   └── qasper_retrieval.py   # per-paper indexes and evaluation
└── results/
    ├── metrics.csv                     # SciFact
    ├── error_analysis.md               # SciFact
    ├── qasper_eval_subset.json
    ├── qasper_retrieval_metrics.csv
    └── qasper_rankings.jsonl
```

Datasets, model caches and embeddings are not committed.

## Running

```bash
pip install -r requirements.txt
```

Both notebooks download their data automatically (SciFact through BEIR, QASPER from the official AllenAI release). A GPU (e.g. the free Colab T4) is recommended for computing embeddings and for reranking.

## Tech stack

Python · Sentence Transformers · Hugging Face · rank-bm25 · BEIR · NumPy · pandas
