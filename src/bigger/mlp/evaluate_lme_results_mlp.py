import json
from pathlib import Path

RESULTS_PATH = Path("outputs/day4_results_mlp_lme.json")


def main():
    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    total = len(results)

    top_raw_session_hit = 0
    top_summary_session_hit = 0
    final_decision_session_hit = 0

    raw_count = 0
    summary_count = 0

    for item in results:
        answer_session_ids = set(str(x) for x in item.get("answer_session_ids", []))

        raw_hit = False
        summary_hit = False

        if item.get("retrieved_chunks"):
            if str(item["retrieved_chunks"][0]["session_id"]) in answer_session_ids:
                raw_hit = True
                top_raw_session_hit += 1

        if item.get("retrieved_summaries"):
            if str(item["retrieved_summaries"][0]["session_id"]) in answer_session_ids:
                summary_hit = True
                top_summary_session_hit += 1

        if item["controller_decision"]["decision"] == "raw":
            raw_count += 1
            if raw_hit:
                final_decision_session_hit += 1
        else:
            summary_count += 1
            if summary_hit:
                final_decision_session_hit += 1

    print("MLP Controller Results")
    print("-" * 50)
    print("Total questions:", total)
    print("Top raw session hit rate:", top_raw_session_hit / total if total else 0.0)
    print("Top summary session hit rate:", top_summary_session_hit / total if total else 0.0)
    print("Final decision session hit rate:", final_decision_session_hit / total if total else 0.0)
    print("Raw decisions:", raw_count)
    print("Summary decisions:", summary_count)


if __name__ == "__main__":
    main()