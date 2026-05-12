import json
import re
from pathlib import Path

INPUT_RESULTS_PATH = Path("outputs/day4_results_wide_deep_lme.json")
OUTPUT_DATASET_PATH = Path("data/processed/lme_controller_dataset_mlp.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def token_set(text):
    return set(re.findall(r"\b\w+\b", normalize_text(text)))


def answer_overlap(a, b):
    a_tokens = token_set(a)
    b_tokens = token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)


def is_precise_question(question: str) -> int:
    q = question.lower()
    keywords = [
        "when", "who", "where", "how long", "how many",
        "what date", "which date", "what year",
        "identity", "quote", "exactly", "sign", "poster",
        "first", "before", "after", "earlier", "later"
    ]
    return int(any(k in q for k in keywords))


def main():
    results = load_json(INPUT_RESULTS_PATH)
    dataset = []

    for item in results:
        question = item["question"]
        answer_session_ids = set(str(x) for x in item.get("answer_session_ids", []))

        retrieved_summaries = item.get("retrieved_summaries", [])
        retrieved_chunks = item.get("retrieved_chunks", [])

        summary_answer = item.get("summary_answer", "")
        raw_answer = item.get("raw_answer", "")

        top_summary_score = retrieved_summaries[0]["score"] if len(retrieved_summaries) > 0 else 0.0
        second_summary_score = retrieved_summaries[1]["score"] if len(retrieved_summaries) > 1 else 0.0

        top_raw_score = retrieved_chunks[0]["score"] if len(retrieved_chunks) > 0 else 0.0
        second_raw_score = retrieved_chunks[1]["score"] if len(retrieved_chunks) > 1 else 0.0

        top_summary_session = str(retrieved_summaries[0]["session_id"]) if len(retrieved_summaries) > 0 else ""
        top_raw_session = str(retrieved_chunks[0]["session_id"]) if len(retrieved_chunks) > 0 else ""

        top_summary_session_order = retrieved_summaries[0]["session_order"] if len(retrieved_summaries) > 0 else -1
        top_raw_session_order = retrieved_chunks[0]["session_order"] if len(retrieved_chunks) > 0 else -1

        summary_hits_answer_session = int(top_summary_session in answer_session_ids) if top_summary_session else 0
        raw_hits_answer_session = int(top_raw_session in answer_session_ids) if top_raw_session else 0

        # Label:
        # 0 = summary, 1 = raw
        if raw_hits_answer_session and not summary_hits_answer_session:
            label = 1
        elif summary_hits_answer_session and not raw_hits_answer_session:
            label = 0
        elif raw_hits_answer_session and summary_hits_answer_session:
            label = 1 if is_precise_question(question) else 0
        else:
            label = 1 if is_precise_question(question) else 0

        num_summary_source_chunks = len(retrieved_summaries[0]["source_chunk_ids"]) if len(retrieved_summaries) > 0 else 0

        summary_is_unknown = int(normalize_text(summary_answer) == "i don't know.")
        raw_is_unknown = int(normalize_text(raw_answer) == "i don't know.")
        answers_exact_match = int(normalize_text(summary_answer) == normalize_text(raw_answer))
        answers_overlap = answer_overlap(summary_answer, raw_answer)

        q_lower = question.lower()

        row = {
            "qa_id": item["qa_id"],
            "sample_id": item["sample_id"],
            "question": question,
            "question_type": item.get("question_type", "unknown"),
            "label": label,

            "top_summary_score": float(top_summary_score),
            "second_summary_score": float(second_summary_score),
            "top_raw_score": float(top_raw_score),
            "second_raw_score": float(second_raw_score),
            "score_gap_raw_minus_summary": float(top_raw_score - top_summary_score),

            "top_summary_session_order": float(top_summary_session_order),
            "top_raw_session_order": float(top_raw_session_order),
            "session_order_gap": float(abs(top_summary_session_order - top_raw_session_order))
            if top_summary_session_order != -1 and top_raw_session_order != -1 else -1.0,

            "num_summary_source_chunks": float(num_summary_source_chunks),

            "question_length": float(len(question.split())),
            "is_when_question": float("when" in q_lower),
            "is_who_question": float("who" in q_lower),
            "is_where_question": float("where" in q_lower),
            "is_how_many_question": float("how many" in q_lower),
            "is_precise_question": float(is_precise_question(question)),

            "summary_is_unknown": float(summary_is_unknown),
            "raw_is_unknown": float(raw_is_unknown),
            "answers_exact_match": float(answers_exact_match),
            "answer_overlap": float(answers_overlap),

            "summary_hits_answer_session": float(summary_hits_answer_session),
            "raw_hits_answer_session": float(raw_hits_answer_session),
        }

        dataset.append(row)

    save_json(dataset, OUTPUT_DATASET_PATH)
    print(f"Saved {len(dataset)} rows to {OUTPUT_DATASET_PATH}")


if __name__ == "__main__":
    main()