import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = Path("data/processed/processed_chunks.json")
SUMMARIES_PATH = Path("data/processed/summaries.json")


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cosine_scores(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    # If embeddings are normalized, dot product == cosine similarity
    return np.dot(doc_embs, query_emb)


class EmbeddingMemoryRetriever:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

        self.chunks = load_json(CHUNKS_PATH)
        self.summaries = load_json(SUMMARIES_PATH)

        self.chunks_by_sample = defaultdict(list)
        self.summaries_by_sample = defaultdict(list)

        for chunk in self.chunks:
            self.chunks_by_sample[chunk["sample_id"]].append(chunk)

        for summary in self.summaries:
            self.summaries_by_sample[summary["sample_id"]].append(summary)

        self.chunk_embeddings = {}
        self.summary_embeddings = {}

        self._build_indices()

    def _build_indices(self):
        print(f"Building embedding indices on device: {self.device}")

        for sample_id, chunks in self.chunks_by_sample.items():
            texts = [
                f"Dialogue chunk:\n{c['text']}"
                for c in chunks
            ]
            emb = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self.chunk_embeddings[sample_id] = emb

        for sample_id, summaries in self.summaries_by_sample.items():
            texts = [
                f"Session summary:\n{s['summary_text']}"
                for s in summaries
            ]
            emb = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            self.summary_embeddings[sample_id] = emb

    def _build_chunk_query(self, question: str) -> str:
        return f"Represent this query for retrieving the exact dialogue chunk that answers it: {question}"

    def _build_summary_query(self, question: str) -> str:
        return f"Represent this query for retrieving the best session summary that can answer it: {question}"

    def retrieve_chunks(self, sample_id: str, question: str, top_k: int = 3) -> List[Dict]:
        chunks = self.chunks_by_sample[sample_id]
        embs = self.chunk_embeddings[sample_id]

        query = self._build_chunk_query(question)
        q_emb = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = cosine_scores(q_emb, embs)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            item = chunks[idx]
            results.append({
                "chunk_id": item["chunk_id"],
                "session_id": item["session_id"],
                "chunk_index": item["chunk_index"],
                "score": float(scores[idx]),
                "text": item["text"],
                "dia_ids": item.get("dia_ids", []),
            })

        return results

    def retrieve_summaries(self, sample_id: str, question: str, top_k: int = 2) -> List[Dict]:
        summaries = self.summaries_by_sample[sample_id]
        embs = self.summary_embeddings[sample_id]

        query = self._build_summary_query(question)
        q_emb = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        scores = cosine_scores(q_emb, embs)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            item = summaries[idx]
            results.append({
                "summary_id": item["summary_id"],
                "session_id": item["session_id"],
                "session_order": item["session_order"],
                "score": float(scores[idx]),
                "summary_text": item["summary_text"],
                "source_chunk_ids": item.get("source_chunk_ids", []),
                "source_dia_ids": item.get("source_dia_ids", []),
            })

        return results