import json
import re
from pathlib import Path
from typing import Dict, Any, List

from chunking import chunk_session_turns

RAW_PATH = Path("data/raw/locomo10.json")
PROCESSED_CHUNKS_PATH = Path("data/processed/processed_chunks.json")
CHUNK_METADATA_PATH = Path("data/processed/chunk_metadata.json")

TURNS_PER_CHUNK = 6


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_sorted_session_ids(conversation: Dict[str, Any]) -> List[str]:
    session_ids = []
    for key in conversation.keys():
        if re.fullmatch(r"session_\d+", key):
            session_ids.append(key)

    session_ids.sort(key=lambda x: int(x.split("_")[1]))
    return session_ids


def main():
    data = load_json(RAW_PATH)

    processed_chunks = []
    chunk_metadata = []

    for record in data:
        sample_id = record["sample_id"]
        conversation = record["conversation"]

        speaker_a = conversation.get("speaker_a")
        speaker_b = conversation.get("speaker_b")

        session_ids = get_sorted_session_ids(conversation)

        for session_id in session_ids:
            session_order = int(session_id.split("_")[1])
            date_time_key = f"{session_id}_date_time"
            date_time = conversation.get(date_time_key)

            turns = conversation.get(session_id)

            # Some records may only have session_X_date_time without actual session_X text
            if turns is None:
                continue

            if not isinstance(turns, list):
                continue

            chunks = chunk_session_turns(
                turns=turns,
                sample_id=sample_id,
                session_id=session_id,
                session_order=session_order,
                date_time=date_time,
                turns_per_chunk=TURNS_PER_CHUNK,
            )

            for chunk in chunks:
                processed_chunks.append({
                    "chunk_id": chunk["chunk_id"],
                    "sample_id": chunk["sample_id"],
                    "session_id": chunk["session_id"],
                    "session_order": chunk["session_order"],
                    "chunk_index": chunk["chunk_index"],
                    "date_time": chunk["date_time"],
                    "dia_ids": chunk["dia_ids"],
                    "text": chunk["text"],
                })

                chunk_metadata.append({
                    "chunk_id": chunk["chunk_id"],
                    "sample_id": chunk["sample_id"],
                    "session_id": chunk["session_id"],
                    "session_order": chunk["session_order"],
                    "chunk_index": chunk["chunk_index"],
                    "start_turn": chunk["start_turn"],
                    "end_turn": chunk["end_turn"],
                    "date_time": chunk["date_time"],
                    "dia_ids": chunk["dia_ids"],
                    "num_chars": len(chunk["text"]),
                    "num_lines": len(chunk["text"].splitlines()),
                    "speaker_a": speaker_a,
                    "speaker_b": speaker_b,
                })

    save_json(processed_chunks, PROCESSED_CHUNKS_PATH)
    save_json(chunk_metadata, CHUNK_METADATA_PATH)

    print(f"Saved {len(processed_chunks)} chunks to {PROCESSED_CHUNKS_PATH}")
    print(f"Saved {len(chunk_metadata)} metadata rows to {CHUNK_METADATA_PATH}")


if __name__ == "__main__":
    main()