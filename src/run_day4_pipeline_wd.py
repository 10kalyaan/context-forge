import json
from pathlib import Path

from embed_retriever import EmbeddingMemoryRetriever
from wide_deep_controller import LearnedWideDeepController
from answerer import answer_from_summaries, answer_from_raw

QA_DATA_PATH = Path("data/processed/qa_dataset.json")
OUTPUT_PATH = Path("outputs/day4_results_wide_deep.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    qa_dataset = load_json(QA_DATA_PATH)
    qa_subset = qa_dataset[:50]

    retriever = EmbeddingMemoryRetriever()
    controller = LearnedWideDeepController()

    results = []

    for qa in qa_subset:
        sample_id = qa["sample_id"]
        question = qa["question"]

        retrieved_summaries = retriever.retrieve_summaries(sample_id, question, top_k=2)
        retrieved_chunks = retriever.retrieve_chunks(sample_id, question, top_k=3)

        summary_answer = answer_from_summaries(question, retrieved_summaries)
        raw_answer = answer_from_raw(question, retrieved_chunks)

        decision = controller.decide(
            question,
            retrieved_summaries,
            retrieved_chunks,
            summary_answer,
            raw_answer
        )

        final_answer = summary_answer if decision["decision"] == "summary" else raw_answer

        results.append({
            "qa_id": qa["qa_id"],
            "sample_id": sample_id,
            "question": question,
            "gold_answer": qa["answer"],
            "retrieved_summaries": retrieved_summaries,
            "retrieved_chunks": retrieved_chunks,
            "summary_answer": summary_answer,
            "raw_answer": raw_answer,
            "controller_decision": decision,
            "final_answer": final_answer,
        })

    save_json(results, OUTPUT_PATH)
    print(f"Saved {len(results)} results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()