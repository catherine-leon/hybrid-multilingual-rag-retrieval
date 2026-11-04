"""Loading and parsing QASPER (raw JSON release v0.3)."""
import json
import random
import re
import tarfile
import urllib.request
from pathlib import Path

QASPER_URLS = {
    "train_dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}
SPLIT_FILES = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}


def download_qasper(data_dir="datasets/qasper"):
    """Download and extract the official QASPER archives (once)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, url in QASPER_URLS.items():
        archive = data_dir / Path(url).name
        if not archive.exists():
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(data_dir)
    return data_dir


def load_qasper_split(split="test", data_dir="datasets/qasper"):
    """Return {paper_id: raw paper dict}."""
    data_dir = Path(data_dir)
    matches = list(data_dir.rglob(SPLIT_FILES[split]))
    if not matches:
        raise FileNotFoundError(f"{SPLIT_FILES[split]} not found under {data_dir}. Run download_qasper().")
    with open(matches[0], encoding="utf-8") as f:
        return json.load(f)


def normalize(text):
    return re.sub(r"\s+", " ", text).strip()


def is_float_evidence(text):
    """Evidence pointing to a figure/table caption, not to a text paragraph."""
    return text.startswith("FLOAT SELECTED")


def paper_paragraphs(paper):
    """Flatten a paper into an ordered list of paragraphs with their section.

    Returns a list of dicts: {"para_idx", "section", "text"}.
    The abstract is included as paragraph 0.
    """
    paragraphs = []
    abstract = normalize(paper.get("abstract", ""))
    if abstract:
        paragraphs.append({"section": "Abstract", "text": abstract})
    for section in paper.get("full_text", []):
        section_name = section.get("section_name") or ""
        for p in section.get("paragraphs", []):
            p = normalize(p)
            if p:
                paragraphs.append({"section": section_name, "text": p})
    for i, p in enumerate(paragraphs):
        p["para_idx"] = i
    return paragraphs


def answer_type(answer):
    if answer.get("unanswerable"):
        return "none"
    if answer.get("extractive_spans"):
        return "extractive"
    if answer.get("free_form_answer"):
        return "abstractive"
    if answer.get("yes_no") is not None:
        return "boolean"
    return "none"


def extract_questions(papers):
    """One record per question, with gold evidence mapped to paragraph indices.

    Evidence from all annotators is united (union). Figure/table evidence
    ("FLOAT SELECTED") is counted separately and not used as retrieval gold.
    """
    records = []
    stats = {"evidence_total": 0, "evidence_float": 0, "evidence_unmatched": 0}

    for paper_id, paper in papers.items():
        paragraphs = paper_paragraphs(paper)
        text_to_idx = {p["text"]: p["para_idx"] for p in paragraphs}

        for qa in paper.get("qas", []):
            gold_paras = set()
            answers = []
            for ann in qa.get("answers", []):
                a = ann["answer"]
                answers.append({
                    "type": answer_type(a),
                    "extractive_spans": a.get("extractive_spans", []),
                    "free_form_answer": a.get("free_form_answer", ""),
                    "yes_no": a.get("yes_no"),
                    "unanswerable": a.get("unanswerable", False),
                })
                for ev in a.get("evidence", []):
                    stats["evidence_total"] += 1
                    ev = normalize(ev)
                    if is_float_evidence(ev):
                        stats["evidence_float"] += 1
                        continue
                    if ev in text_to_idx:
                        gold_paras.add(text_to_idx[ev])
                    else:
                        stats["evidence_unmatched"] += 1

            records.append({
                "paper_id": paper_id,
                "question_id": qa["question_id"],
                "question": qa["question"],
                "gold_paragraphs": sorted(gold_paras),
                "answers": answers,
            })
    return records, stats


def sample_eval_subset(records, n=300, seed=42, path=None):
    """Fixed evaluation subset: questions that have text evidence.

    If `path` exists, the saved subset is reused, so every experiment
    runs on exactly the same questions.
    """
    if path is not None and Path(path).exists():
        ids = set(json.loads(Path(path).read_text()))
        return [r for r in records if r["question_id"] in ids]

    with_evidence = [r for r in records if r["gold_paragraphs"]]
    rng = random.Random(seed)
    subset = rng.sample(with_evidence, min(n, len(with_evidence)))
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([r["question_id"] for r in subset], indent=1))
    return subset
