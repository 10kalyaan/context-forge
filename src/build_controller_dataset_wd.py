import json
import re
from pathlib import Path

DAY3_RESULTS_PATH = Path("outputs/day3_results.json")
OUTPUT_JSON_PATH = Path("data/processed/controller_dataset_wd.json")


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
    if isinstance(text, bool):
        text = str(text)
    elif not isinstance(text, str):
        text = str(text)
    return re.sub(r"\s+", " ", text.strip().lower())


def token_set(text):
    return set(re.findall(r"\b\w+\b", normalize_text(text)))


def simple_match_score(pred, gold):
    pred = normalize_text(pred)
    gold = normalize_text(gold)

    if not pred or pred == "i don't know.":
        return 0.0

    if pred == gold:
        return 1.0

    if gold in pred or pred in gold:
        return 0.7

    pred_tokens = token_set(pred)
    gold_tokens = token_set(gold)

    if not gold_tokens:
        return 0.0

    return len(pred_tokens & gold_tokens) / max(len(gold_tokens), 1)


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
        "identity", "quote", "exactly", "sign", "poster"
    ]
    return int(any(k in q for k in keywords))


def main():
    results = load_json(DAY3_RESULTS_PATH)
    dataset = []

    for item in results:
        question = item["question"]
        gold = item.get("gold_answer", "") or ""

        retrieved_summaries = item.get("retrieved_summaries", [])
        retrieved_chunks = item.get("retrieved_chunks", [])

        summary_answer = item.get("summary_answer", "")
        raw_answer = item.get("raw_answer", "")

        summary_score = simple_match_score(summary_answer, gold)
        raw_score = simple_match_score(raw_answer, gold)

        label = 1 if raw_score >= summary_score else 0
        # 1 = raw, 0 = summary

        top_summary_score = retrieved_summaries[0]["score"] if len(retrieved_summaries) > 0 else 0.0
        second_summary_score = retrieved_summaries[1]["score"] if len(retrieved_summaries) > 1 else 0.0

        top_raw_score = retrieved_chunks[0]["score"] if len(retrieved_chunks) > 0 else 0.0
        second_raw_score = retrieved_chunks[1]["score"] if len(retrieved_chunks) > 1 else 0.0

        top_summary_session_order = retrieved_summaries[0]["session_order"] if len(retrieved_summaries) > 0 else -1

        if len(retrieved_chunks) > 0:
            raw_session_id = retrieved_chunks[0]["session_id"]
            top_raw_session_order = int(raw_session_id.split("_")[1])
        else:
            top_raw_session_order = -1

        num_summary_source_chunks = len(retrieved_summaries[0]["source_chunk_ids"]) if len(retrieved_summaries) > 0 else 0
        num_summary_source_dia_ids = len(retrieved_summaries[0]["source_dia_ids"]) if len(retrieved_summaries) > 0 else 0

        q_lower = question.lower()

        summary_is_unknown = int(normalize_text(summary_answer) == "i don't know.")
        raw_is_unknown = int(normalize_text(raw_answer) == "i don't know.")
        answers_exact_match = int(normalize_text(summary_answer) == normalize_text(raw_answer))
        answer_token_overlap = answer_overlap(summary_answer, raw_answer)

        same_session_top_summary_top_raw = int(
            top_summary_session_order != -1 and
            top_raw_session_order != -1 and
            top_summary_session_order == top_raw_session_order
        )

        row = {
            "qa_id": item["qa_id"],
            "sample_id": item["sample_id"],
            "question": question,
            "gold_answer": gold,
            "summary_answer": summary_answer,
            "raw_answer": raw_answer,
            "label": label,

            "top_summary_score": top_summary_score,
            "second_summary_score": second_summary_score,
            "top_raw_score": top_raw_score,
            "second_raw_score": second_raw_score,
            "score_gap_raw_minus_summary": top_raw_score - top_summary_score,

            "num_summary_source_chunks": num_summary_source_chunks,
            "num_summary_source_dia_ids": num_summary_source_dia_ids,

            "question_length": len(question.split()),
            "is_when_question": int("when" in q_lower),
            "is_who_question": int("who" in q_lower),
            "is_where_question": int("where" in q_lower),
            "is_how_many_question": int("how many" in q_lower),
            "is_precise_question": is_precise_question(question),

            "top_summary_session_order": top_summary_session_order,
            "top_raw_session_order": top_raw_session_order,
            "session_order_gap": abs(top_summary_session_order - top_raw_session_order)
            if top_summary_session_order != -1 and top_raw_session_order != -1 else -1,
            "same_session_top_summary_top_raw": same_session_top_summary_top_raw,

            "summary_is_unknown": summary_is_unknown,
            "raw_is_unknown": raw_is_unknown,
            "answers_exact_match": answers_exact_match,
            "answer_token_overlap": answer_token_overlap,
        }

        dataset.append(row)

    save_json(dataset, OUTPUT_JSON_PATH)
    print(f"Saved {len(dataset)} rows to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()