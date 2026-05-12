
import json
from pathlib import Path

RAW_PATH = Path("data/raw/longmemeval/longmemeval_oracle.json")
QA_PATH = Path("data/processed/lme_qa_dataset.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    data = load_json(RAW_PATH)

    qa_data = []
    for record in data:
        qa_data.append({
            "qa_id": record["question_id"],
            "sample_id": record["question_id"],
            "question_id": record["question_id"],
            "question_type": record["question_type"],
            "question": record["question"],
            "answer": record["answer"],
            "question_date": record["question_date"],
            "answer_session_ids": record.get("answer_session_ids", []),
        })

    save_json(qa_data, QA_PATH)
    print(f"Saved {len(qa_data)} QA rows to {QA_PATH}")


if __name__ == "__main__":
    main()