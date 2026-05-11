import json
from pathlib import Path

RESULTS_PATH = Path("outputs/day3_results.json")

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

print("num results:", len(results))

first = results[0]

print("\nQUESTION:")
print(first["question"])

print("\nGOLD ANSWER:")
print(first["gold_answer"])

print("\nCONTROLLER DECISION:")
print(first["controller_decision"])

print("\nTOP SUMMARY:")
print(first["retrieved_summaries"][0])

print("\nTOP RAW CHUNK:")
print(first["retrieved_chunks"][0])

print("\nSUMMARY ANSWER:")
print(first["summary_answer"])

print("\nRAW ANSWER:")
print(first["raw_answer"])

print("\nFINAL ANSWER:")
print(first["final_answer"])