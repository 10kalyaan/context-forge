from typing import List, Dict, Any


def format_turn(turn: Dict[str, Any]) -> str:
    speaker = turn["speaker"]
    text = turn["text"]
    return f"{speaker}: {text}"


def chunk_session_turns(
    turns: List[Dict[str, Any]],
    sample_id: str,
    session_id: str,
    session_order: int,
    date_time: str | None,
    turns_per_chunk: int = 6,
) -> List[Dict[str, Any]]:
    chunks = []

    for i in range(0, len(turns), turns_per_chunk):
        sub_turns = turns[i:i + turns_per_chunk]
        chunk_index = i // turns_per_chunk
        start_turn = i
        end_turn = i + len(sub_turns) - 1

        chunk_text = "\n".join(format_turn(t) for t in sub_turns)

        chunk_id = f"{sample_id}_{session_id}_chunk_{chunk_index:04d}"

        dia_ids = [t["dia_id"] for t in sub_turns]

        chunks.append({
            "chunk_id": chunk_id,
            "sample_id": sample_id,
            "session_id": session_id,
            "session_order": session_order,
            "chunk_index": chunk_index,
            "start_turn": start_turn,
            "end_turn": end_turn,
            "date_time": date_time,
            "dia_ids": dia_ids,
            "text": chunk_text,
        })

    return chunks