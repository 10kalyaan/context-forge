import json
from pathlib import Path

RAW_PATH = Path("data/raw/longmemeval/longmemeval_oracle.json")
SUMMARIES_PATH = Path("data/processed/lme_summaries.json")
SUMMARY_MAP_PATH = Path("data/processed/lme_summary_to_raw_map.json")


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


def make_session_summary(session_turns):
    if len(session_turns) <= 4:
        selected = session_turns
    else:
        selected = session_turns[:2] + session_turns[-2:]
    return " ".join(format_turn(t) for t in selected)


def main():
    data = load_json(RAW_PATH)

    summaries = []
    summary_to_raw_map = []

    for record in data:
        sample_id = record["question_id"]
        session_ids = record["haystack_session_ids"]
        session_dates = record["haystack_dates"]
        haystack_sessions = record["haystack_sessions"]

        for i, (session_id, date_time, session_turns) in enumerate(zip(session_ids, session_dates, haystack_sessions)):
            session_order = i + 1
            session_id = str(session_id)

            summary_id = f"{sample_id}_{session_id}_summary"
            summary_text = make_session_summary(session_turns)

            # chunk ids must match preprocess_longmemeval.py logic
            n_chunks = (len(session_turns) + 5) // 6
            source_chunk_ids = [
                f"{sample_id}_{session_id}_chunk_{chunk_idx:04d}"
                for chunk_idx in range(n_chunks)
            ]

            summaries.append({
                "summary_id": summary_id,
                "sample_id": sample_id,
                "session_id": session_id,
                "session_order": session_order,
                "date_time": date_time,
                "summary_text": summary_text,
                "source_chunk_ids": source_chunk_ids,
            })

            summary_to_raw_map.append({
                "summary_id": summary_id,
                "sample_id": sample_id,
                "session_id": session_id,
                "source_chunk_ids": source_chunk_ids,
            })

    save_json(summaries, SUMMARIES_PATH)
    save_json(summary_to_raw_map, SUMMARY_MAP_PATH)

    print(f"Saved {len(summaries)} summaries to {SUMMARIES_PATH}")
    print(f"Saved {len(summary_to_raw_map)} mappings to {SUMMARY_MAP_PATH}")


if __name__ == "__main__":
    main()