import torch
import torch.nn as nn
from pathlib import Path


MODEL_PATH = Path("outputs/wide_deep_controller.pt")

WIDE_KEYS = [
    "is_when_question",
    "is_who_question",
    "is_where_question",
    "is_how_many_question",
    "is_precise_question",
    "same_session_top_summary_top_raw",
    "summary_is_unknown",
    "raw_is_unknown",
    "answers_exact_match",
]

DEEP_KEYS = [
    "top_summary_score",
    "second_summary_score",
    "top_raw_score",
    "second_raw_score",
    "score_gap_raw_minus_summary",
    "num_summary_source_chunks",
    "num_summary_source_dia_ids",
    "question_length",
    "top_summary_session_order",
    "top_raw_session_order",
    "session_order_gap",
    "answer_token_overlap",
]


class WideDeepController(nn.Module):
    def __init__(self, wide_dim, deep_dim):
        super().__init__()

        self.deep = nn.Sequential(
            nn.Linear(deep_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(32 + wide_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, wide_x, deep_x):
        deep_out = self.deep(deep_x)
        combined = torch.cat([wide_x, deep_out], dim=1)
        return self.final(combined)


def normalize_text(text):
    if text is None:
        return ""
    return " ".join(text.strip().lower().split())


def answer_overlap(a, b):
    a_tokens = set(normalize_text(a).split())
    b_tokens = set(normalize_text(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)


def is_precise_question(question: str) -> int:
    q = question.lower()
    keywords = [
        "when", "who", "where", "how long", "how many",
        "what date", "which date", "what year",
        "identity", "quote", "exactly", "sign", "poster"
    ]
    return int(any(k in q for k in keywords))


def extract_features(question, retrieved_summaries, retrieved_chunks, summary_answer, raw_answer):
    q_lower = question.lower()

    top_summary_score = retrieved_summaries[0]["score"] if len(retrieved_summaries) > 0 else 0.0
    second_summary_score = retrieved_summaries[1]["score"] if len(retrieved_summaries) > 1 else 0.0

    top_raw_score = retrieved_chunks[0]["score"] if len(retrieved_chunks) > 0 else 0.0
    second_raw_score = retrieved_chunks[1]["score"] if len(retrieved_chunks) > 1 else 0.0

    top_summary_session_order = retrieved_summaries[0]["session_order"] if len(retrieved_summaries) > 0 else -1
    if len(retrieved_chunks) > 0:
        top_raw_session_order = int(retrieved_chunks[0]["session_id"].split("_")[1])
    else:
        top_raw_session_order = -1

    num_summary_source_chunks = len(retrieved_summaries[0]["source_chunk_ids"]) if len(retrieved_summaries) > 0 else 0
    num_summary_source_dia_ids = len(retrieved_summaries[0]["source_dia_ids"]) if len(retrieved_summaries) > 0 else 0

    summary_is_unknown = int(normalize_text(summary_answer) == "i don't know.")
    raw_is_unknown = int(normalize_text(raw_answer) == "i don't know.")
    answers_exact_match = int(normalize_text(summary_answer) == normalize_text(raw_answer))
    same_session_top_summary_top_raw = int(
        top_summary_session_order != -1 and
        top_raw_session_order != -1 and
        top_summary_session_order == top_raw_session_order
    )

    wide_feats = [
        float("when" in q_lower),
        float("who" in q_lower),
        float("where" in q_lower),
        float("how many" in q_lower),
        float(is_precise_question(question)),
        float(same_session_top_summary_top_raw),
        float(summary_is_unknown),
        float(raw_is_unknown),
        float(answers_exact_match),
    ]

    deep_feats = [
        float(top_summary_score),
        float(second_summary_score),
        float(top_raw_score),
        float(second_raw_score),
        float(top_raw_score - top_summary_score),
        float(num_summary_source_chunks),
        float(num_summary_source_dia_ids),
        float(len(question.split())),
        float(top_summary_session_order),
        float(top_raw_session_order),
        float(abs(top_summary_session_order - top_raw_session_order))
        if top_summary_session_order != -1 and top_raw_session_order != -1 else -1.0,
        float(answer_overlap(summary_answer, raw_answer)),
    ]

    return wide_feats, deep_feats


class LearnedWideDeepController:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)

        self.model = WideDeepController(
            wide_dim=len(WIDE_KEYS),
            deep_dim=len(DEEP_KEYS)
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def decide(self, question, retrieved_summaries, retrieved_chunks, summary_answer, raw_answer):
        wide_feats, deep_feats = extract_features(
            question, retrieved_summaries, retrieved_chunks, summary_answer, raw_answer
        )

        wide_x = torch.tensor([wide_feats], dtype=torch.float32).to(self.device)
        deep_x = torch.tensor([deep_feats], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(wide_x, deep_x)
            pred = torch.argmax(logits, dim=1).item()

        if pred == 1:
            return {"decision": "raw", "reason": "wide_deep_controller"}
        return {"decision": "summary", "reason": "wide_deep_controller"}