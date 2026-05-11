import json
from pathlib import Path

SUMMARIES_PATH = Path("data/processed/summaries.json")
CHUNKS_PATH = Path("data/processed/processed_chunks.json")

with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
    summaries = json.load(f)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

# check the first 3 summaries
for idx, sample_summary in enumerate(summaries[:3], start=1):
    print("=" * 100)
    print(f"SUMMARY {idx}")
    print("SUMMARY ID:", sample_summary["summary_id"])
    print("SAMPLE ID:", sample_summary["sample_id"])
    print("SESSION ID:", sample_summary["session_id"])
    print("DATE TIME:", sample_summary["date_time"])

    print("\nSUMMARY TEXT:\n")
    print(sample_summary["summary_text"])

    print("\nSOURCE CHUNKS:\n")
    for chunk_id in sample_summary["source_chunk_ids"]:
        print("-" * 80)
        print("CHUNK ID:", chunk_id)
        print(chunks_by_id[chunk_id]["text"])
    print("\n")