import json
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/locomo10.json")
QA_DATA_PATH = Path("data/processed/qa_dataset.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    raw_data = load_json(RAW_DATA_PATH)
    qa_dataset = []

    for record in raw_data:
        sample_id = record["sample_id"]
        qa_list = record.get("qa", [])

        for idx, qa in enumerate(qa_list):
            qa_dataset.append({
                "qa_id": f"{sample_id}_qa_{idx:04d}",
                "sample_id": sample_id,
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "adversarial_answer": qa.get("adversarial_answer"),
                "category": qa.get("category"),
                "evidence": qa.get("evidence", []),
            })

    save_json(qa_dataset, QA_DATA_PATH)
    print(f"Saved {len(qa_dataset)} QA items to {QA_DATA_PATH}")


if __name__ == "__main__":
    main()