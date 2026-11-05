# Hybrid and Cross-lingual Retrieval for Scientific RAG

## Motivation and research questions

A retrieval-augmented generation (RAG) system can only ground its answer in evidence that the retriever actually finds. Retrieval methods are usually compared on corpus-level IR benchmarks, where the task is to find relevant documents in a large collection. Question answering over a scientific paper poses a different problem. The evidence has to be located *inside* one long document, under a limited context budget, and the user's question may be in a different language than the paper.

This project studies how retrieval design choices behave in that setting:

| | Research question | Stage | Status |
|---|---|---|---|
| **RQ1** | Do conclusions about sparse, dense, hybrid and reranked retrieval drawn on a corpus-level benchmark transfer to evidence retrieval within scientific papers? | 1–2 | ✅ |
| **RQ2** | How does the chunking strategy affect evidence coverage at a fixed context budget? | 2 | ✅ |
| **RQ3** | Does better evidence retrieval translate into more accurate and more faithful generated answers? | 3 | ✅ |
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

The evaluation subset contains 300 questions with text evidence from 221 papers, sampled with a fixed seed (`results/qasper_eval_subset.json`). The same subset is reused in all later stages.

### Chunking strategies

Both retrieval models read at most 512 tokens, and longer inputs are silently truncated. Chunk sizes are therefore measured in tokens of the `multilingual-e5-base` tokenizer and capped at 256 tokens:

| Strategy | Description | Chunks / paper | Mean tokens |
|---|---|---:|---:|
| paragraph | one chunk per paragraph; paragraphs longer than 256 tokens are split inside the paragraph, and all pieces keep the paragraph id | 52.6 | 112 |
| paragraph+section | same, with the section name prepended (counted in the budget), e.g. *"Experiments ::: Datasets. …"* | 53.0 | 119 |
| fixed256 | 256-token windows with a 64-token overlap, ignoring paragraph and section boundaries | 30.2 | 253 |

Splitting matters more than its overall frequency suggests. Only 6.9% of paragraphs exceed 256 tokens (748 / 10,802; the longest has 5,301), yet 11.0% of gold evidence paragraphs do (84 / 762). Long paragraphs are more likely to contain the evidence, and without splitting they would be truncated by the models.

Chunk sizes differ between strategies, so the strategies are compared with:
- **ParaRecall@k**: the share of gold paragraphs covered by the top-k chunks;
- **Tokens@k**: the size of the context that would be passed to the LLM.

Differences are tested with a **paired bootstrap** over the 300 questions (1,000 resamples, 95% CI).

### Results

| Chunking | Method | ParaRecall@1 | ParaRecall@3 | ParaRecall@5 | MRR@5 | Tokens@5 |
|---|---|---:|---:|---:|---:|---:|
| paragraph | BM25 | 0.113 | 0.291 | 0.377 | 0.316 | 736 |
| paragraph | Dense | 0.198 | 0.398 | 0.508 | 0.474 | 694 |
| paragraph | Hybrid | 0.161 | 0.351 | 0.474 | 0.434 | 748 |
| paragraph | Hybrid+CE | 0.207 | 0.418 | 0.539 | 0.494 | 777 |
| paragraph+section | BM25 | 0.121 | 0.286 | 0.386 | 0.324 | 755 |
| paragraph+section | Dense | 0.196 | 0.387 | 0.527 | 0.477 | 732 |
| paragraph+section | Hybrid | 0.165 | 0.365 | 0.491 | 0.443 | 781 |
| paragraph+section | **Hybrid+CE** | **0.209** | **0.432** | **0.564** | **0.498** | 790 |
| fixed256 | BM25 | 0.171 | 0.384 | 0.544 | 0.406 | 1,268 |
| fixed256 | Dense | 0.245 | 0.486 | 0.652 | 0.506 | 1,263 |
| fixed256 | Hybrid | 0.228 | 0.480 | 0.652 | 0.495 | 1,266 |
| fixed256 | **Hybrid+CE** | **0.285** | **0.554** | **0.680** | **0.561** | 1,269 |

Full metrics: `results/qasper_retrieval_metrics.csv`.

| Paired bootstrap, ParaRecall@5 | Δ (B − A) | 95% CI | Significant |
|---|---:|---|:---:|
| paragraph: BM25 → Dense | +0.131 | [0.086, 0.179] | yes |
| paragraph: Dense → Hybrid | −0.033 | [−0.066, 0.002] | no |
| paragraph: Hybrid → Hybrid+CE | +0.065 | [0.022, 0.105] | yes |
| paragraph: Dense → Hybrid+CE | +0.032 | [−0.008, 0.069] | no |
| BM25: paragraph → paragraph+section | +0.009 | [−0.011, 0.030] | no |
| Hybrid+CE: paragraph → paragraph+section | +0.025 | [0.001, 0.051] | yes |
| Hybrid+CE: paragraph+section → fixed256 (≈1.6× more tokens) | +0.116 | [0.078, 0.156] | yes |

`results/qasper_bootstrap.csv`

### RQ1: do the SciFact conclusions transfer?

ParaRecall@5 with paragraph chunks is the paragraph-level counterpart of Recall@5 on SciFact.

| Difference at k = 5 | SciFact (Recall@5) | QASPER, paragraph (ParaRecall@5) |
|---|---:|---:|
| Dense − BM25 | +0.032 | **+0.131** (significant) |
| Hybrid − Dense | −0.017 | −0.033 (not significant) |
| Hybrid+CE − Hybrid | +0.044 | **+0.065** (significant) |

- **The direction of every effect is the same in both settings.**
- **The gap between lexical and semantic retrieval is about four times larger on QASPER.** A likely reason: QASPER questions were written without access to the paper's full text, so they rarely reuse its wording.
- **Rank fusion does not beat Dense alone at k = 5 in either setting.** On QASPER the loss is not statistically significant.
- **Reranking significantly improves Hybrid, but Hybrid+CE is not significantly better than Dense alone** (+0.032, CI [−0.008, 0.069]). For evidence retrieval inside a paper, a single dense retriever is a strong and much cheaper baseline.

The two settings differ in both query type and search space, so this comparison shows consistency of the effects, not their cause.

### RQ2: chunking at a fixed context budget

At equal k, fixed windows look much better (+0.116 for Hybrid+CE), but they pass about 1.6 times more tokens. Compared at a similar budget of about 730–790 tokens:

| Method | paragraph+section @5 | tokens | fixed256 @3 | tokens |
|---|---:|---:|---:|---:|
| BM25 | 0.386 | 755 | 0.384 | 762 |
| Dense | 0.527 | 732 | 0.486 | 757 |
| Hybrid | 0.491 | 781 | 0.480 | 759 |
| Hybrid+CE | 0.564 | 790 | 0.554 | 763 |

- **At a comparable budget, paragraph-aware chunking matches or slightly exceeds fixed windows** (differences of 0.002–0.041; not tested for significance). The apparent advantage of fixed windows at equal k comes from the larger context.
- **The section name helps the reranked pipeline slightly** (+0.025, significant but close to the limit) **and has no measurable effect on BM25.**
- **The best configuration at this budget (paragraph+section, Hybrid+CE) covers 56% of the gold evidence.** This motivates adaptive retrieval (RQ5).

### Error analysis

| Error pattern (top 5, paragraph chunks) | Questions (of 300) |
|---|---:|
| BM25 fails, Dense succeeds | 71 |
| Dense succeeds, Hybrid fails | 28 |
| Hybrid fails, Hybrid+CE succeeds | 42 |
| Section name makes BM25 succeed | 11 |

| Pattern | Question | Gold evidence (excerpt) | Behaviour |
|---|---|---|---|
| Lexical mismatch | *How many languages are in the dataset?* | *"This consisted of 76 documents in English and 47 in Spanish"* | BM25: not in top 5. Dense: rank 1. |
| Fusion hurts, reranking repairs | *By how much did their model outperform baselines?* | *"Our initial submits RUN 4 and 5 outperformed the memorization baseline on every metric on every task"* | Dense: rank 1. Hybrid: not in top 5. Hybrid+CE: rank 2. |
| Reranking | *What language is this dataset in?* | *"The dataset makes use of 450 English-language news sites"* | Hybrid: not in top 5. Hybrid+CE: rank 1. |
| Relevant but unannotated chunk | *What domain of text are they working with?* | gold: *"we retrieve hundreds of thousands of news articles … from the DailyMail website"* (Dense: rank 5) | Dense's top-1 chunk, *"In a nutshell, we retrieved hundreds of thousands of news articles … from DailyMail"*, says the same thing in another paragraph but is not annotated |

In the last case, a paragraph that answers the question is counted as a miss because only another paragraph was annotated. This suggests that the evidence annotation is not exhaustive, and the scores may underestimate retrieval quality.

More examples: `results/qasper_error_analysis.txt`.

## Stage 3: grounded answer generation (RQ3)

### Setup

The design was fixed before the test run:

| | |
|---|---|
| Questions | the same 300 test questions as in stage 2 |
| Generator | `Qwen/Qwen2.5-7B-Instruct`, bfloat16, greedy decoding (temperature 0), one prompt for all conditions, answers always in English |
| Context | top-5 chunks from stage 2 (paragraph+section chunking), capped at 1,280 tokens |
| Output | JSON with a short answer and the numbers of the excerpts used, mapped back to chunk ids |
| Conditions | **closed-book** (no context, lower bound), **BM25**, **Dense**, **Hybrid**, **Hybrid+CE**, **oracle** (chunks with gold evidence, upper bound) |
| Metrics | Answer-F1 and Evidence-F1 (QASPER metrics, maximum over annotators), citation rate and precision, refusal rate; paired bootstrap |

**Prompt development on the dev split only.** The prompt was checked on 30 dev questions with gold context, and the test subset was not used. Version 1 often copied whole sentences, which lowers token F1, so version 2 asks for a short phrase. Overall F1 did not change beyond noise, but the answers moved to the QASPER answer style. Version 2 was frozen before the test run.

| Dev check (30 questions) | v1 | v2 (used) |
|---|---:|---:|
| oracle Answer-F1 | 0.486 | 0.473 |
| — extractive | 0.589 | 0.609 |
| — abstractive | 0.287 | 0.312 |
| closed-book Answer-F1 | 0.077 | 0.118 |
| closed-book refusals ("Unanswerable") | 93% | 67% |
| valid JSON | 100% | 100% |

### Results

| Condition | Answer-F1 | Evidence-F1 | Gold evidence in context | Citation rate | Citation precision | "Unanswerable" | Context tokens | Gap closed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| closed-book | 0.131 | 0.220 | — | 0% | — | 78.7% | 0 | 0% |
| BM25 | 0.386 | 0.387 | 57.3% | 85.0% | 0.427 | 23.7% | 755 | 54% |
| Dense | **0.456** | 0.493 | 73.7% | 90.7% | 0.520 | 18.0% | 732 | **69%** |
| Hybrid | 0.429 | 0.471 | 69.3% | 89.3% | 0.507 | 18.0% | 781 | 63% |
| Hybrid+CE | **0.453** | **0.497** | 76.0% | 90.3% | 0.513 | 13.7% | 790 | **68%** |
| oracle | 0.603 | 0.898 | 100% | 92.0% | 1.000 | 12.7% | 372 | 100% |

- *Evidence-F1*: overlap between the paragraphs the model cites and each annotator's evidence paragraphs (maximum over annotators). An empty citation list scores 1.0 against an annotator who gave no evidence, which is why closed-book is not at 0.
- *Citation rate*: share of answers with at least one citation. *Citation precision*: share of cited chunks that contain gold evidence, computed only over answers with citations.
- *Gap closed* = (F1 − closed-book F1) / (oracle F1 − closed-book F1).

**Cross-check with the official evaluator.** Running `qasper_evaluator.py` from the QASPER release (text evidence only) on the same 300 questions gives identical Answer-F1, Answer-F1 by type and Evidence-F1 for all six conditions (`results/official_eval_*.txt`).

| Paired bootstrap, Answer-F1 | Δ (B − A) | 95% CI | Significant |
|---|---:|---|:---:|
| closed-book → Hybrid+CE | +0.322 | [0.275, 0.374] | yes |
| BM25 → Dense | +0.070 | [0.029, 0.115] | yes |
| Dense → Hybrid | −0.027 | [−0.059, 0.003] | no |
| Hybrid → Hybrid+CE | +0.025 | [−0.008, 0.059] | no |
| Dense → Hybrid+CE | −0.003 | [−0.037, 0.032] | no |
| Hybrid+CE → oracle | +0.150 | [0.109, 0.187] | yes |

| Answer-F1 by answer type (best-matching annotator, official grouping) | extractive | abstractive | yes / no | unanswerable |
|---|---:|---:|---:|---:|
| closed-book | 0.037 | 0.042 | 0.091 | 0.931 |
| BM25 | 0.372 | 0.197 | 0.632 | 0.846 |
| Dense | 0.467 | 0.224 | 0.632 | 0.800 |
| Hybrid | 0.449 | 0.218 | 0.564 | 0.714 |
| Hybrid+CE | 0.465 | 0.231 | 0.632 | 0.846 |
| oracle | 0.633 | 0.349 | 0.737 | 0.923 |

The type is taken from the annotator whose answer matches the prediction best, so the number of questions per type differs slightly between conditions (for closed-book: 166 / 72 / 33 / 29). The *unanswerable* column contains questions where at least one annotator found no answer, although another gave evidence. Abstractive answers are the hardest even with oracle context (0.349), because a free-form answer rarely shares many tokens with the reference.

### RQ3: does better retrieval give better answers?

| Retriever | F1, gold evidence in context | n | F1, no gold evidence | n |
|---|---:|---:|---:|---:|
| BM25 | 0.458 | 172 | 0.288 | 128 |
| Dense | 0.517 | 221 | 0.285 | 79 |
| Hybrid | 0.512 | 208 | 0.240 | 92 |
| Hybrid+CE | 0.504 | 228 | 0.294 | 72 |

- **Retrieval drives answer quality.** With at least one gold paragraph in the context, Answer-F1 is about 0.50–0.52; without one, it is about 0.24–0.29. As in stage 2, BM25 is the weakest retriever and is significantly worse than Dense on answers (−0.070), and Hybrid stays below Dense.
- **Small retrieval gains do not reach the answers.** Hybrid+CE retrieved slightly more evidence than Dense in stage 2 (+0.032, not significant), and at the answer level the two are identical (−0.003). Question by question, Hybrid+CE is better on 61, worse on 59 and equal on 180. **A single dense retriever is as good as the full hybrid + reranking pipeline, at a fraction of the cost.**
- **Answers without gold evidence are not all wrong** (F1 ≈ 0.29). This agrees with the stage-2 observation that relevant but unannotated paragraphs are often retrieved.
- **Better retrieval reduces refusals** (BM25 23.7% → Hybrid+CE 13.7% "Unanswerable").
- **The generator is a bottleneck of similar size.** Even with oracle context, F1 is 0.603, and 91 of 300 answers score below 0.3. For Hybrid+CE, 72 questions (24%) fail at retrieval (no gold evidence in context), and 93 (31%) score below 0.3 although gold evidence was in the context.

### Error analysis

| Type | Question | Gold | Hybrid+CE | oracle |
|---|---|---|---|---|
| Retrieval failure | *What language is explored in this paper?* | English | Tweet content | English |
| Retrieval failure → refusal | *What domain of text are they working with?* | news articles | Unanswerable | news |
| Partially correct, low F1 | *What is the architecture of the encoder?* | pre-trained Xnlg with a 10-layer encoder | Transformer | 10-layer |
| Correct gist, low F1 | *What kind of evaluations do use to evaluate dialogue?* | perplexity, BLEU, ACUTE-EVA | both automatic and human evaluation | both automatic and human evaluation |
| Generation error | *Does LadaBERT ever outperform its knowledge destilation teacher in terms of accuracy on some problems?* | No | Yes | Unanswerable |

The third and fourth rows show the limits of token F1: answers that are partly or broadly right but phrased differently score low. Comparisons between conditions remain valid, since all conditions use the same metric. The absolute scores, however, underestimate answer quality.

More examples: `results/generation_error_analysis.txt`.

### Limitations of this stage

- **Split gold paragraphs.** When a long gold paragraph is split into several chunks, every piece counts as gold evidence (QASPER evidence is paragraph-level). A retrieved piece may therefore miss the exact sentence with the answer, which slightly blurs the with / without evidence comparison. This concerns 84 of 762 gold paragraphs (11%).
- **No unanswerable detection.** The subset contains only questions with text evidence, so the "Unanswerable" column is a refusal rate, not an evaluation of unanswerable-question detection.
- The oracle context is shorter (372 tokens on average) because only gold chunks are included. It is an upper bound on context quality, not on context size.
- The results use one generator and one prompt, and token F1 does not capture paraphrases.

## Next steps

| Stage | Research question | Plan |
|---|---|---|
| 3b. Second metric | RQ3 | An LLM judge (a different, larger model) rates correctness and faithfulness to the context on ~100 questions for Dense, Hybrid+CE and oracle. This checks whether the conclusions hold beyond token F1. |
| 4. Cross-lingual | RQ4 | The 300 questions are machine-translated into French and Russian. Comparison: direct multilingual retrieval vs translating the question back into English with a different MT system. Answers are generated in English in all conditions, so answer F1 stays comparable to the English baseline. |
| 5. Adaptive retrieval | RQ5 | Evidence-sufficiency check → query rewriting / translation → retry. Static vs always-retry vs adaptive at the same final context budget: answer quality, evidence recall, number of retries, latency. |

## Repository structure

```text
├── notebooks/
│   ├── 01_scifact_retrieval_benchmark.ipynb
│   └── 02_qasper_chunking_retrieval.ipynb
├── scripts/
│   ├── run_stage2.py         # stage 2: chunking + retrieval
│   └── run_stage3.py         # stage 3: dev prompt check, generation, evaluation
├── src/
│   ├── retrieval.py          # BM25, dense, RRF, reranking (shared)
│   ├── evaluation.py         # Recall@k, MRR@k, error-case search
│   ├── qasper.py             # QASPER loading, evidence → paragraph mapping, eval subset
│   ├── chunking.py           # token-based chunking strategies + gold chunk mapping
│   ├── qasper_retrieval.py   # per-paper indexes, evaluation, paired bootstrap
│   ├── pipeline.py           # rebuilds the stage-2 data for later stages
│   ├── generation.py         # context budget, prompt, JSON parsing, LLM backends, resumable runs
│   └── answer_eval.py        # Answer-F1, citation precision
└── results/
    ├── metrics.csv, error_analysis.md          # stage 1 (SciFact)
    ├── qasper_eval_subset.json                 # the 300 test questions
    ├── qasper_*.csv / .txt / .jsonl            # stage 2
    └── generation_*.csv / .txt / .jsonl,
        predictions_*.jsonl                     # stage 3
```

Datasets, model caches and embeddings are not committed.

## Running

```bash
pip install -r requirements.txt
python scripts/run_stage2.py                 # retrieval (or notebook 02)
python scripts/run_stage3.py --phase dev     # prompt check on the dev split
python scripts/run_stage3.py --phase test    # generation + evaluation on the test subset
```

QASPER and the models are downloaded automatically. Stages 2 and 3 were run on an NVIDIA RTX A6000 (about 10 and 30 minutes). Stage 3 needs about 16 GB of GPU memory; use `--load-in-4bit` on smaller GPUs.

## Tech stack

Python · PyTorch · Hugging Face Transformers · Sentence Transformers · rank-bm25 · BEIR · NumPy · pandas
