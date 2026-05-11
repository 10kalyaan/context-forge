import json
from pathlib import Path

from embed_retriever import EmbeddingMemoryRetriever

QA_DATA_PATH = Path("data/processed/qa_dataset.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    qa_dataset = load_json(QA_DATA_PATH)
    retriever = EmbeddingMemoryRetriever()

    test_items = qa_dataset[:5]

    for item in test_items:
        print("=" * 100)
        print("QUESTION:", item["question"])
        print("GOLD ANSWER:", item["answer"])
        print("EVIDENCE:", item["evidence"])

        summaries = retriever.retrieve_summaries(item["sample_id"], item["question"], top_k=2)
        chunks = retriever.retrieve_chunks(item["sample_id"], item["question"], top_k=3)

        print("\nTOP SUMMARIES:")
        for s in summaries:
            print(f"- {s['summary_id']} | score={s['score']:.4f}")
            print(s["summary_text"][:250], "...\n")

        print("TOP CHUNKS:")
        for c in chunks:
            print(f"- {c['chunk_id']} | score={c['score']:.4f} | dia_ids={c['dia_ids']}")
            print(c["text"][:250], "...\n")


if __name__ == "__main__":
    main()