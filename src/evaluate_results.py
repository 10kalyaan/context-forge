import json
import re
from pathlib import Path

DAY3_PATH = Path("outputs/day3_results.json")
DAY4_WD_PATH = Path("outputs/day4_results_wide_deep.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def simple_match(pred, gold):
    pred = normalize_text(pred)
    gold = normalize_text(gold)

    if not pred or pred == "i don't know.":
        return 0

    if pred == gold:
        return 1

    if gold in pred or pred in gold:
        return 1

    return 0


def evaluate(results, name):
    total = len(results)
    correct = 0
    raw_count = 0
    summary_count = 0

    for item in results:
        if simple_match(item.get("final_answer"), item.get("gold_answer")):
            correct += 1

        decision = item.get("controller_decision", {}).get("decision")
        if decision == "raw":
            raw_count += 1
        else:
            summary_count += 1

    print(f"\n{name}")
    print("-" * 40)
    print("Total:", total)
    print("Correct:", correct)
    print("Accuracy:", correct / total if total else 0.0)
    print("Raw decisions:", raw_count)
    print("Summary decisions:", summary_count)


def main():
    day3 = load_json(DAY3_PATH)
    day4_wd = load_json(DAY4_WD_PATH)

    evaluate(day3, "Day 3 Rule-Based Controller")
    evaluate(day4_wd, "Day 4 Wide & Deep Controller")


if __name__ == "__main__":
    main()