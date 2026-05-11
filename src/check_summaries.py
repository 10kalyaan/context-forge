import json
from pathlib import Path

SUMMARIES_PATH = Path("data/processed/summaries.json")
SUMMARY_TO_RAW_MAP_PATH = Path("data/processed/summary_to_raw_map.json")

with open(SUMMARIES_PATH, "r", encoding="utf-8") as f:
    summaries = json.load(f)

with open(SUMMARY_TO_RAW_MAP_PATH, "r", encoding="utf-8") as f:
    mappings = json.load(f)

print("num summaries:", len(summaries))

print("\nfirst summary:")
print(json.dumps(summaries[0], indent=2, ensure_ascii=False))

print("\nfirst mapping:")
print(json.dumps(mappings[0], indent=2, ensure_ascii=False))