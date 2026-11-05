"""Rebuild the stage-2 data (papers, eval subset, chunks, gold) in one call.

Used by later notebooks so they work on exactly the same questions and chunks
as stage 2, without copying its cells. CPU only, about a minute.
"""
from src.qasper import load_qasper_split, paper_paragraphs, extract_questions, sample_eval_subset
from src.chunking import make_chunkers, gold_chunk_ids


def prepare_qasper(root, tokenizer, max_tokens=256, split="test", n=300, seed=42, subset_file="qasper_eval_subset.json"):
    """split="dev" with another subset_file gives a separate prompt-development set."""
    papers = load_qasper_split(split, root / "datasets" / "qasper")
    records, _ = extract_questions(papers)
    eval_qs = sample_eval_subset(records, n=n, seed=seed, path=root / "results" / subset_file)
    eval_papers = sorted({r["paper_id"] for r in eval_qs})
    paragraphs_by_paper = {pid: paper_paragraphs(papers[pid]) for pid in eval_papers}

    chunks, chunk_info, gold = {}, {}, {}
    for name, chunker in make_chunkers(tokenizer, max_tokens).items():
        by_paper = {pid: chunker(pid, paragraphs_by_paper[pid]) for pid in eval_papers}
        chunks[name] = by_paper
        chunk_info[name] = {c["chunk_id"]: c for cs in by_paper.values() for c in cs}
        gold[name] = {r["question_id"]: gold_chunk_ids(by_paper[r["paper_id"]], r["gold_paragraphs"])
                      for r in eval_qs}
    return {"papers": papers, "eval_qs": eval_qs, "chunks": chunks, "chunk_info": chunk_info,
            "gold": gold, "paragraphs": paragraphs_by_paper}
