from collections import defaultdict
from utils import load_json

chunks = load_json("data/processed/processed_chunks.json")
meta = load_json("data/processed/chunk_metadata.json")

print(f"Loaded {len(chunks)} chunks")
print(f"Loaded {len(meta)} metadata rows")

chunks_by_sample = defaultdict(list)
for c in chunks:
    chunks_by_sample[c["sample_id"]].append(c)

print(f"Number of conversations: {len(chunks_by_sample)}")

first_sample = next(iter(chunks_by_sample))
print(f"\nFirst sample_id: {first_sample}")
print(f"Number of chunks in first sample: {len(chunks_by_sample[first_sample])}")

print("\nFirst chunk:")
print(chunks_by_sample[first_sample][0])