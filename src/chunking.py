"""Chunking strategies for QASPER papers.

Every chunk keeps the list of paragraph indices it covers ("para_ids"),
so gold evidence (annotated at paragraph level) can be mapped to chunks
whatever the strategy.
"""


def make_chunk(paper_id, n, text, section, para_ids):
    return {
        "chunk_id": f"{paper_id}::{n}",
        "paper_id": paper_id,
        "section": section,
        "text": text,
        "para_ids": sorted(set(para_ids)),
    }


def paragraph_chunks(paper_id, paragraphs, add_section=False):
    """Strategy A: one chunk = one native paragraph.

    add_section=True prepends the section name ("section-aware" variant),
    e.g. "Experiments ::: Datasets. We evaluate on ..."
    """
    chunks = []
    for n, p in enumerate(paragraphs):
        text = f"{p['section']}. {p['text']}" if add_section and p["section"] else p["text"]
        chunks.append(make_chunk(paper_id, n, text, p["section"], [p["para_idx"]]))
    return chunks


def fixed_size_chunks(paper_id, paragraphs, chunk_size=200, overlap=50):
    """Strategy B: fixed windows of `chunk_size` words with `overlap`,
    ignoring paragraph and section boundaries.

    Words are whitespace tokens (simple and model-independent);
    200 words stay well below the 512-token limit of E5 / bge-reranker.
    """
    assert 0 <= overlap < chunk_size
    words, word_para, word_section = [], [], []
    for p in paragraphs:
        for w in p["text"].split():
            words.append(w)
            word_para.append(p["para_idx"])
            word_section.append(p["section"])

    chunks, step, n = [], chunk_size - overlap, 0
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunks.append(make_chunk(
            paper_id, n,
            " ".join(words[start:end]),
            word_section[start],
            word_para[start:end],
        ))
        n += 1
        if end == len(words):
            break
    return chunks


def gold_chunk_ids(chunks, gold_paragraphs):
    """A chunk is relevant if it contains at least part of a gold paragraph."""
    gold = set(gold_paragraphs)
    return [c["chunk_id"] for c in chunks if gold & set(c["para_ids"])]


CHUNKERS = {
    "paragraph": lambda pid, paras: paragraph_chunks(pid, paras, add_section=False),
    "paragraph+section": lambda pid, paras: paragraph_chunks(pid, paras, add_section=True),
    "fixed200": lambda pid, paras: fixed_size_chunks(pid, paras, 200, 50),
}
