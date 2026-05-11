import json
from pathlib import Path
from collections import defaultdict

RAW_DATA_PATH = Path("data/raw/locomo10.json")
PROCESSED_CHUNKS_PATH = Path("data/processed/processed_chunks.json")
SUMMARIES_PATH = Path("data/processed/summaries.json")
SUMMARY_TO_RAW_MAP_PATH = Path("data/processed/summary_to_raw_map.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    raw_data = load_json(RAW_DATA_PATH)
    processed_chunks = load_json(PROCESSED_CHUNKS_PATH)

    # Group raw chunks by (sample_id, session_id)
    chunks_by_session = defaultdict(list)
    for chunk in processed_chunks:
        key = (chunk["sample_id"], chunk["session_id"])
        chunks_by_session[key].append(chunk)

    # Sort chunk order within each session
    for key in chunks_by_session:
        chunks_by_session[key].sort(key=lambda x: x["chunk_index"])

    summaries = []
    summary_to_raw_map = []

    for record in raw_data:
        sample_id = record["sample_id"]
        conversation = record["conversation"]
        session_summary = record.get("session_summary", {})

        for summary_key, summary_text in session_summary.items():
            # Example summary_key: session_1_summary
            if not summary_key.startswith("session_") or not summary_key.endswith("_summary"):
                continue

            session_id = summary_key.replace("_summary", "")
            session_order = int(session_id.split("_")[1])
            date_time = conversation.get(f"{session_id}_date_time")

            session_chunks = chunks_by_session.get((sample_id, session_id), [])
            if not session_chunks:
                continue

            summary_id = f"{sample_id}_{session_id}_summary"

            source_chunk_ids = [chunk["chunk_id"] for chunk in session_chunks]

            source_dia_ids = []
            for chunk in session_chunks:
                source_dia_ids.extend(chunk.get("dia_ids", []))

            summaries.append({
                "summary_id": summary_id,
                "sample_id": sample_id,
                "session_id": session_id,
                "session_order": session_order,
                "date_time": date_time,
                "summary_text": summary_text,
                "source_chunk_ids": source_chunk_ids,
                "source_dia_ids": source_dia_ids,
            })

            summary_to_raw_map.append({
                "summary_id": summary_id,
                "sample_id": sample_id,
                "session_id": session_id,
                "source_chunk_ids": source_chunk_ids,
                "source_dia_ids": source_dia_ids,
            })

    save_json(summaries, SUMMARIES_PATH)
    save_json(summary_to_raw_map, SUMMARY_TO_RAW_MAP_PATH)

    print(f"Saved {len(summaries)} summaries to {SUMMARIES_PATH}")
    print(f"Saved {len(summary_to_raw_map)} summary mappings to {SUMMARY_TO_RAW_MAP_PATH}")


if __name__ == "__main__":
    main()