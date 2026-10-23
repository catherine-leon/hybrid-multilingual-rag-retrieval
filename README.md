# Hybrid and Multilingual Retrieval for RAG

> **Project status: ongoing**

This project explores retrieval strategies for RAG through three stages:

1. **SciFact retrieval benchmark — completed**  
   Comparison of BM25, dense retrieval, hybrid RRF, and Cross-Encoder reranking using Recall@k, MRR@k, and qualitative error analysis.

2. **Multilingual retrieval — planned**  
   Extension of the same pipeline to English, French, and Russian to compare retrieval robustness across languages.

3. **Agentic retrieval — planned**  
   Reuse of the retrieval components in an adaptive RAG pipeline with retrieval routing, query rewriting, retries, and evidence verification.

The repository currently contains the completed **SciFact benchmark**, which serves as the baseline for the next stages.

## Overview

Retrieval quality is a critical component of Retrieval-Augmented Generation: if relevant evidence is not retrieved or is ranked too low, the generation model cannot reliably ground its answer.

The current benchmark evaluates how different retrieval strategies affect:

- retrieval coverage;
- top-rank relevance;
- robustness to lexical vs semantic mismatch;
- candidate quality before generation.

The pipeline compares:

1. **BM25** — sparse lexical retrieval;
2. **Dense retrieval** — multilingual sentence embeddings;
3. **Hybrid retrieval** — BM25 and Dense combined with Reciprocal Rank Fusion;
4. **Cross-Encoder reranking** — relevance reranking of the hybrid candidate set.

## Pipeline

```text
Query
  |
  |---- BM25 ------------------|
  |                            |
  |---- Dense Retrieval -------|
                               |
                    Reciprocal Rank Fusion
                               |
                        Top-20 candidates
                               |
                         Cross-Encoder
                               |
                          Final ranking
```

## Dataset

The first stage uses **SciFact** in the **BEIR** benchmark format.

SciFact contains:

- scientific claims used as queries;
- scientific abstracts used as retrieval documents;
- expert relevance annotations identifying relevant evidence.

All retrieval methods are evaluated on the same corpus, queries, relevance labels, and metrics.

## Methods

### BM25

BM25 serves as the lexical baseline.

It ranks documents based on term overlap while accounting for term frequency, inverse document frequency, document length, and term-frequency saturation.

### Dense Retrieval

Dense retrieval uses:

`intfloat/multilingual-e5-base`

Queries and documents are independently encoded into a shared embedding space.

E5-specific prefixes are used:

```text
query: ...
passage: ...
```

Document and query embeddings are normalized, allowing cosine similarity to be computed using the dot product.

### Hybrid Retrieval

Sparse and dense rankings are combined using **Reciprocal Rank Fusion (RRF)**:

```text
RRF(d) = sum 1 / (k + rank(d))
```

with `k = 60`.

The top 100 results from each retriever are used as fusion candidates.

The goal is to combine:

- BM25's strength in exact terminology and entity matching;
- dense retrieval's ability to capture semantic similarity and paraphrases.

### Cross-Encoder Reranking

The hybrid candidate set is reranked with:

`BAAI/bge-reranker-v2-m3`

Unlike dense retrieval, the Cross-Encoder processes the query and document jointly and directly predicts their relevance.

Because this is computationally more expensive, reranking is applied only to the top 20 hybrid candidates.

## Evaluation

The following metrics are used:

- **Recall@1**
- **Recall@5**
- **Recall@10**
- **MRR@1**
- **MRR@5**
- **MRR@10**

Recall measures how much of the gold evidence appears within the top-k retrieved documents.

Mean Reciprocal Rank measures how highly the first relevant document appears in the ranking.

For each query, retrieval is performed once at the maximum required depth, and all metrics are derived from the same ranked list.

## Results

| Method | Recall@1 | Recall@5 | Recall@10 | MRR@1 | MRR@5 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.508 | 0.728 | 0.774 | 0.523 | 0.613 | 0.619 |
| Dense | 0.557 | 0.760 | 0.807 | 0.583 | 0.660 | 0.666 |
| Hybrid RRF | 0.563 | 0.743 | 0.833 | 0.583 | 0.657 | 0.668 |
| **Hybrid + Cross-Encoder** | **0.613** | **0.787** | **0.836** | **0.643** | **0.710** | **0.715** |

## Key Findings

### Dense retrieval improves over the lexical baseline

Dense retrieval outperforms BM25 across all reported metrics, showing the benefit of semantic matching when query and document vocabulary differ.

### Hybrid retrieval improves candidate coverage

Hybrid RRF achieves a higher Recall@10 than either BM25 or Dense alone.

This suggests that lexical and semantic retrieval recover partially complementary relevant documents.

### Cross-Encoder reranking improves top-rank quality

Compared with Hybrid RRF:

- Recall@1 increases from **0.563 to 0.613**;
- MRR@1 increases from **0.583 to 0.643**;
- MRR@5 increases from **0.657 to 0.710**.

Recall@10 changes only slightly because reranking primarily changes the order of documents already present in the candidate set.

Across the SciFact queries, the Cross-Encoder moved a gold document to a higher position in **67 cases**.

## Qualitative Error Analysis

Aggregate metrics do not fully explain why one retrieval method outperforms another. Representative cases were therefore examined manually.

### 1. BM25 fails, Dense succeeds

**Query ID:** 239  
**Gold document:** 14079881

> *Cellular aging closely links to an older appearance.*

BM25 retrieves documents containing literal *cellular aging* terminology but misses the gold document in the top 10.

Dense retrieval ranks the gold document **3rd**, successfully connecting *older appearance / perceived age* with biological and molecular ageing.

This illustrates the advantage of semantic retrieval when the query and relevant document use different vocabulary.

### 2. Dense fails, BM25 succeeds

**Query ID:** 72  
**Gold document:** 6076903

> *Activator-inhibitor pairs are provided dorsally by Admpchordin.*

BM25 retrieves the gold document at **rank 9**, whereas Dense misses it in the top 10.

The relevant paper explicitly discusses **ADMP**, **Chordin**, and their opposing signaling relationship.

This case shows that exact domain-specific terminology can still favor lexical retrieval.

### 3. Dense fails, Hybrid succeeds

**Query ID:** 70  
**Gold documents:** 4414547, 5956380

> *Activation of PPM1D suppresses p53 function.*

Dense retrieval returns documents related to general p53 regulation but misses both gold documents.

Hybrid retrieval combines semantic and lexical evidence and retrieves both PPM1D-specific gold documents at **ranks 2 and 6**.

This demonstrates how rank fusion can recover entity-specific evidence missed by dense retrieval alone.

### 4. Cross-Encoder improves Hybrid ranking

**Query ID:** 185  
**Gold document:** 18340282

> *Breast cancer development is determined exclusively by genetic factors.*

Hybrid retrieval identifies the relevant study:

*Gene-environment interactions in 7610 women with breast cancer: prospective evidence from the Million Women Study*

but ranks it only **6th**.

The Cross-Encoder moves the document to **rank 1**.

The study directly examines genetic susceptibility together with established environmental risk factors, making it a strong semantic match for the claim.

This illustrates the main role of reranking: the relevant evidence is already present in the candidate set, but joint query-document modeling improves its position substantially.

## Repository Structure

```text
hybrid-multilingual-rag-retrieval/
|
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- notebooks/
|   `-- 01_scifact_retrieval_benchmark.ipynb
|
|-- src/
|   |-- retrieval.py
|   `-- evaluation.py
|
`-- results/
    |-- metrics.csv
    `-- error_analysis.md
```

The notebook contains the experimental workflow and analysis, while reusable retrieval and evaluation functions are stored in `src/`.

Datasets, model caches, and generated embeddings are not committed to the repository.

## Installation

Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd hybrid-multilingual-rag-retrieval
pip install -r requirements.txt
```

The SciFact dataset can be downloaded automatically through BEIR.

Document embeddings are computed once and can be cached locally for subsequent runs.

## Next Steps

The next stage will extend the retrieval benchmark to a multilingual setting with **English, French, and Russian**.

The goal is to compare the relative behavior of BM25, dense retrieval, hybrid fusion, and reranking across languages.

A later extension will reuse the same retrieval components in an adaptive agentic RAG pipeline.

Possible components include:

- retrieval strategy selection;
- query rewriting;
- retrieval retries;
- evidence sufficiency checks;
- multilingual routing;
- grounding verification.

## Tech Stack

- Python
- PyTorch
- Hugging Face
- Sentence Transformers
- BM25
- Cross-Encoder reranking
- BEIR
- NumPy
- pandas
