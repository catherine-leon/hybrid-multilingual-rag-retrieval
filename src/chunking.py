"""Token-based chunking strategies for QASPER papers.

Sizes are measured with the tokenizer of the dense retriever, so that no chunk
is silently truncated by the retrieval models (E5 / bge-reranker: 512 tokens).

Every chunk keeps the list of paragraph indices it covers ("para_ids"), so gold
evidence (annotated at paragraph level) can be mapped to chunks whatever the
strategy.
"""


def make_chunk(paper_id, n, text, section, para_ids, n_tokens):
    return {
        "chunk_id": f"{paper_id}::{n}",
        "paper_id": paper_id,
        "section": section,
        "text": text,
        "para_ids": sorted(set(para_ids)),
        "n_tokens": n_tokens,
    }


def token_spans(tokenizer, text):
    """Character spans (start, end) of the tokens of `text`, no special tokens."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [(s, e) for s, e in enc["offset_mapping"] if e > s]


def windows(n, size, overlap):
    """Start/end indices of sliding windows over n items."""
    assert 0 <= overlap < size
    if n <= size:
        return [(0, n)]
    out, start = [], 0
    while True:
        end = min(start + size, n)
        out.append((start, end))
        if end == n:
            return out
        start += size - overlap


def paragraph_chunks(paper_id, paragraphs, tokenizer, max_tokens=256, overlap=32, add_section=False):
    """Paragraph-aware chunking.

    A paragraph that fits in `max_tokens` is one chunk. A longer paragraph is
    split INSIDE the paragraph into overlapping windows; all pieces keep the
    same para_id. add_section=True prepends the section name, and its tokens
    count towards the budget.
    """
    chunks, n = [], 0
    for p in paragraphs:
        prefix = f"{p['section']}. " if add_section and p["section"] else ""
        budget = max_tokens - (len(token_spans(tokenizer, prefix)) if prefix else 0)
        spans = token_spans(tokenizer, p["text"])
        for s, e in windows(len(spans), budget, overlap):
            body = p["text"][spans[s][0]:spans[e - 1][1]]
            chunks.append(make_chunk(paper_id, n, prefix + body, p["section"],
                                     [p["para_idx"]], (e - s) + (max_tokens - budget)))
            n += 1
    return chunks


def fixed_token_chunks(paper_id, paragraphs, tokenizer, size=256, overlap=64):
    """Fixed windows of `size` tokens with `overlap`, ignoring paragraph and
    section boundaries. Text is rebuilt from the original paragraphs, so a
    window spanning two paragraphs joins their pieces with a space."""
    toks = []  # (para position, char start, char end)
    for i, p in enumerate(paragraphs):
        toks += [(i, s, e) for s, e in token_spans(tokenizer, p["text"])]

    chunks = []
    for n, (s, e) in enumerate(windows(len(toks), size, overlap)):
        pieces, order = {}, []
        for i, cs, ce in toks[s:e]:
            if i not in pieces:
                pieces[i] = [cs, ce]
                order.append(i)
            pieces[i][1] = ce
        text = " ".join(paragraphs[i]["text"][pieces[i][0]:pieces[i][1]] for i in order)
        chunks.append(make_chunk(paper_id, n, text, paragraphs[order[0]]["section"],
                                 [paragraphs[i]["para_idx"] for i in order], e - s))
    return chunks


def gold_chunk_ids(chunks, gold_paragraphs):
    """A chunk is relevant if it contains at least part of a gold paragraph."""
    gold = set(gold_paragraphs)
    return [c["chunk_id"] for c in chunks if gold & set(c["para_ids"])]


def make_chunkers(tokenizer, max_tokens=256):
    return {
        "paragraph": lambda pid, paras: paragraph_chunks(pid, paras, tokenizer, max_tokens),
        "paragraph+section": lambda pid, paras: paragraph_chunks(pid, paras, tokenizer, max_tokens, add_section=True),
        f"fixed{max_tokens}": lambda pid, paras: fixed_token_chunks(pid, paras, tokenizer, max_tokens, 64),
    }
