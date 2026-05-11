import json
from pathlib import Path

chunks_path = Path("data/processed/processed_chunks.json")
meta_path = Path("data/processed/chunk_metadata.json")

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

with open(meta_path, "r", encoding="utf-8") as f:
    meta = json.load(f)

print("num chunks:", len(chunks))
print("\nfirst chunk:")
print(json.dumps(chunks[0], indent=2, ensure_ascii=False))

print("\nfirst metadata row:")
print(json.dumps(meta[0], indent=2, ensure_ascii=False))