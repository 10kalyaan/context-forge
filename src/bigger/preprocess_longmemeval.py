import json
from pathlib import Path

RAW_PATH = Path("data/raw/longmemeval/longmemeval_oracle.json")
PROCESSED_CHUNKS_PATH = Path("data/processed/lme_processed_chunks.json")
CHUNK_METADATA_PATH = Path("data/processed/lme_chunk_metadata.json")

TURNS_PER_CHUNK = 6


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def format_turn(turn):
    role = turn.get("role", "unknown")
    content = turn.get("content", "")
    return f"{role}: {content}"


def main():
    data = load_json(RAW_PATH)

    processed_chunks = []
    chunk_metadata = []

    for record in data:
        sample_id = record["question_id"]
        session_ids = record["haystack_session_ids"]
        session_dates = record["haystack_dates"]
        haystack_sessions = record["haystack_sessions"]

        for i, (session_id, date_time, session_turns) in enumerate(zip(session_ids, session_dates, haystack_sessions)):
            session_order = i + 1

            for j in range(0, len(session_turns), TURNS_PER_CHUNK):
                sub_turns = session_turns[j:j + TURNS_PER_CHUNK]
                chunk_index = j // TURNS_PER_CHUNK
                start_turn = j
                end_turn = j + len(sub_turns) - 1

                chunk_id = f"{sample_id}_{session_id}_chunk_{chunk_index:04d}"
                text = "\n".join(format_turn(t) for t in sub_turns)

                turn_answer_flags = [bool(t.get("has_answer", False)) for t in sub_turns]

                processed_chunks.append({
                    "chunk_id": chunk_id,
                    "sample_id": sample_id,
                    "session_id": str(session_id),
                    "session_order": session_order,
                    "chunk_index": chunk_index,
                    "date_time": date_time,
                    "text": text,
                    "has_answer_flags": turn_answer_flags,
                })

                chunk_metadata.append({
                    "chunk_id": chunk_id,
                    "sample_id": sample_id,
                    "session_id": str(session_id),
                    "session_order": session_order,
                    "chunk_index": chunk_index,
                    "start_turn": start_turn,
                    "end_turn": end_turn,
                    "date_time": date_time,
                    "num_chars": len(text),
                    "num_lines": len(text.splitlines()),
                    "contains_answer_turn": int(any(turn_answer_flags)),
                })

    save_json(processed_chunks, PROCESSED_CHUNKS_PATH)
    save_json(chunk_metadata, CHUNK_METADATA_PATH)

    print(f"Saved {len(processed_chunks)} chunks to {PROCESSED_CHUNKS_PATH}")
    print(f"Saved {len(chunk_metadata)} metadata rows to {CHUNK_METADATA_PATH}")


if __name__ == "__main__":
    main()