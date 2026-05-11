from typing import List, Dict


def is_precise_question(question: str) -> bool:
    q = question.lower()
    keywords = [
        "when",
        "how long",
        "how many",
        "what date",
        "which date",
        "what year",
        "what was the sign",
        "what did the poster say",
        "who",
        "where",
        "identity",
        "exactly",
        "quote"
    ]
    return any(k in q for k in keywords)


def decide_memory_source(
    question: str,
    retrieved_summaries: List[Dict],
    retrieved_chunks: List[Dict]
) -> Dict:
    if not retrieved_chunks:
        return {"decision": "summary", "reason": "no_raw_retrieved"}

    if not retrieved_summaries:
        return {"decision": "raw", "reason": "no_summary_retrieved"}

    top_summary_score = retrieved_summaries[0]["score"]
    top_raw_score = retrieved_chunks[0]["score"]

    if is_precise_question(question):
        return {"decision": "raw", "reason": "precise_question_type"}

    if top_raw_score >= top_summary_score:
        return {"decision": "raw", "reason": "raw_score_higher_or_equal"}

    if top_summary_score < 0.78:
        return {"decision": "raw", "reason": "summary_score_too_low"}

    return {"decision": "summary", "reason": "summary_sufficient"}