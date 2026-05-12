import re
from typing import List, Dict


def build_summary_context(retrieved_summaries: List[Dict]) -> str:
    parts = []
    for idx, item in enumerate(retrieved_summaries, start=1):
        parts.append(f"[Summary {idx}] {item['summary_text']}")
    return "\n\n".join(parts)


def build_raw_context(retrieved_chunks: List[Dict]) -> str:
    parts = []
    for idx, item in enumerate(retrieved_chunks, start=1):
        parts.append(f"[Chunk {idx}] {item['text']}")
    return "\n\n".join(parts)


def split_into_units(context: str) -> List[str]:
    """
    Split context into candidate answer units.
    For summaries this will usually be sentences.
    For raw chunks it may also include dialogue lines.
    """
    candidates = []
    for block in context.split("\n"):
        block = block.strip()
        if not block:
            continue

        parts = re.split(r'(?<=[.!?])\s+', block)
        for p in parts:
            p = p.strip()
            if p:
                candidates.append(p)

    return candidates


def tokenize(text: str) -> set:
    return set(re.findall(r"\b\w+\b", text.lower()))


def lexical_overlap_score(question: str, candidate: str) -> float:
    q_tokens = tokenize(question)
    c_tokens = tokenize(candidate)

    if not q_tokens or not c_tokens:
        return 0.0

    overlap = len(q_tokens & c_tokens)
    return overlap / max(len(q_tokens), 1)


def extract_best_span(question: str, context: str) -> str:
    candidates = split_into_units(context)

    if not candidates:
        return "I don't know."

    scored = []
    for cand in candidates:
        score = lexical_overlap_score(question, cand)
        scored.append((cand, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    best_text, best_score = scored[0]

    if best_score < 0.10:
        return "I don't know."

    return best_text


def answer_from_summaries(question: str, retrieved_summaries: List[Dict]) -> str:
    context = build_summary_context(retrieved_summaries)
    return extract_best_span(question, context)


def answer_from_raw(question: str, retrieved_chunks: List[Dict]) -> str:
    context = build_raw_context(retrieved_chunks)
    return extract_best_span(question, context)