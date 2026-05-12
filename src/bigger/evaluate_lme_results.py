import json
from pathlib import Path

RESULTS_PATH = Path("outputs/day4_results_wide_deep_lme.json")

with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    results = json.load(f)

total = len(results)

top_raw_session_hit = 0
top_summary_session_hit = 0
final_decision_session_hit = 0

raw_count = 0
summary_count = 0

question_type_stats = {}

for item in results:
    answer_session_ids = set(str(x) for x in item.get("answer_session_ids", []))
    qtype = item.get("question_type", "unknown")

    if qtype not in question_type_stats:
        question_type_stats[qtype] = {
            "total": 0,
            "raw_hit": 0,
            "summary_hit": 0,
            "final_hit": 0,
        }

    question_type_stats[qtype]["total"] += 1

    raw_hit = False
    summary_hit = False
    final_hit = False

    if item.get("retrieved_chunks"):
        top_raw_session = str(item["retrieved_chunks"][0]["session_id"])
        if top_raw_session in answer_session_ids:
            raw_hit = True
            top_raw_session_hit += 1
            question_type_stats[qtype]["raw_hit"] += 1

    if item.get("retrieved_summaries"):
        top_summary_session = str(item["retrieved_summaries"][0]["session_id"])
        if top_summary_session in answer_session_ids:
            summary_hit = True
            top_summary_session_hit += 1
            question_type_stats[qtype]["summary_hit"] += 1

    decision = item["controller_decision"]["decision"]
    if decision == "raw":
        raw_count += 1
        final_hit = raw_hit
    else:
        summary_count += 1
        final_hit = summary_hit

    if final_hit:
        final_decision_session_hit += 1
        question_type_stats[qtype]["final_hit"] += 1

print("Overall Metrics")
print("-" * 50)
print("Total questions:", total)
print("Top raw session hit rate:", top_raw_session_hit / total if total else 0.0)
print("Top summary session hit rate:", top_summary_session_hit / total if total else 0.0)
print("Final decision session hit rate:", final_decision_session_hit / total if total else 0.0)
print("Raw decisions:", raw_count)
print("Summary decisions:", summary_count)

print("\nPer Question Type")
print("-" * 50)
for qtype, stats in question_type_stats.items():
    t = stats["total"]
    print(f"\n{qtype}")
    print("  Total:", t)
    print("  Raw session hit rate:", stats["raw_hit"] / t if t else 0.0)
    print("  Summary session hit rate:", stats["summary_hit"] / t if t else 0.0)
    print("  Final decision session hit rate:", stats["final_hit"] / t if t else 0.0)