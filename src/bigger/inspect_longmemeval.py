import json
from pathlib import Path
from pprint import pprint

DATA_PATH = Path("data/raw/longmemeval/longmemeval_oracle.json")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Top-level type:", type(data))
print("Num records:", len(data))

first = data[0]
print("\nTop-level keys:")
print(first.keys())

print("\nFirst record:")
pprint(first)

print("\nFirst haystack session id:", first["haystack_session_ids"][0])
print("First haystack date:", first["haystack_dates"][0])
print("Type of first haystack session:", type(first["haystack_sessions"][0]))
print("Length of first haystack session:", len(first["haystack_sessions"][0]))
print("\nFirst turn in first session:")
pprint(first["haystack_sessions"][0][0])