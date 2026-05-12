import json
from pathlib import Path

RESULTS_PATH = Path("outputs/day4_results_wide_deep_lme.json")

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

print("num results:", len(results))

for i, item in enumerate(results[:5], start=1):
    print("=" * 100)
    print(f"RESULT {i}")
    print("QUESTION:", item["question"])
    print("QUESTION TYPE:", item["question_type"])
    print("GOLD ANSWER:", item["gold_answer"])
    print("ANSWER SESSION IDS:", item["answer_session_ids"])
    print("CONTROLLER DECISION:", item["controller_decision"])

    if item["retrieved_summaries"]:
        print("\nTOP SUMMARY:")
        print(item["retrieved_summaries"][0])

    if item["retrieved_chunks"]:
        print("\nTOP RAW CHUNK:")
        print(item["retrieved_chunks"][0])

    print("\nSUMMARY ANSWER:")
    print(item["summary_answer"])

    print("\nRAW ANSWER:")
    print(item["raw_answer"])

    print("\nFINAL ANSWER:")
    print(item["final_answer"])
    print()