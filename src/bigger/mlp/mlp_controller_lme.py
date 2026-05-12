import re
from pathlib import Path

import torch
import torch.nn as nn

MODEL_PATH = Path("outputs/mlp_controller_lme.pt")

FEATURE_KEYS = [
    "top_summary_score",
    "second_summary_score",
    "top_raw_score",
    "second_raw_score",
    "score_gap_raw_minus_summary",
    "top_summary_session_order",
    "top_raw_session_order",
    "session_order_gap",
    "num_summary_source_chunks",
    "question_length",
    "is_when_question",
    "is_who_question",
    "is_where_question",
    "is_how_many_question",
    "is_precise_question",
    "summary_is_unknown",
    "raw_is_unknown",
    "answers_exact_match",
    "answer_overlap",
]


class MLPController(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)


def normalize_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def token_set(text):
    return set(re.findall(r"\b\w+\b", normalize_text(text)))


def answer_overlap(a, b):
    a_tokens = token_set(a)
    b_tokens = token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)


def is_precise_question(question: str) -> int:
    q = question.lower()
    keywords = [
        "when", "who", "where", "how long", "how many",
        "what date", "which date", "what year",
        "identity", "quote", "exactly", "sign", "poster",
        "first", "before", "after", "earlier", "later"
    ]
    return int(any(k in q for k in keywords))


def extract_features(question, retrieved_summaries, retrieved_chunks, summary_answer, raw_answer):
    q_lower = question.lower()

    top_summary_score = retrieved_summaries[0]["score"] if len(retrieved_summaries) > 0 else 0.0
    second_summary_score = retrieved_summaries[1]["score"] if len(retrieved_summaries) > 1 else 0.0

    top_raw_score = retrieved_chunks[0]["score"] if len(retrieved_chunks) > 0 else 0.0
    second_raw_score = retrieved_chunks[1]["score"] if len(retrieved_chunks) > 1 else 0.0

    top_summary_session_order = retrieved_summaries[0]["session_order"] if len(retrieved_summaries) > 0 else -1
    top_raw_session_order = retrieved_chunks[0]["session_order"] if len(retrieved_chunks) > 0 else -1

    session_order_gap = abs(top_summary_session_order - top_raw_session_order) \
        if top_summary_session_order != -1 and top_raw_session_order != -1 else -1.0

    num_summary_source_chunks = len(retrieved_summaries[0]["source_chunk_ids"]) if len(retrieved_summaries) > 0 else 0

    summary_is_unknown = int(normalize_text(summary_answer) == "i don't know.")
    raw_is_unknown = int(normalize_text(raw_answer) == "i don't know.")
    answers_exact_match = int(normalize_text(summary_answer) == normalize_text(raw_answer))

    feats = [
        float(top_summary_score),
        float(second_summary_score),
        float(top_raw_score),
        float(second_raw_score),
        float(top_raw_score - top_summary_score),
        float(top_summary_session_order),
        float(top_raw_session_order),
        float(session_order_gap),
        float(num_summary_source_chunks),
        float(len(question.split())),
        float("when" in q_lower),
        float("who" in q_lower),
        float("where" in q_lower),
        float("how many" in q_lower),
        float(is_precise_question(question)),
        float(summary_is_unknown),
        float(raw_is_unknown),
        float(answers_exact_match),
        float(answer_overlap(summary_answer, raw_answer)),
    ]

    return feats


class LearnedMLPController:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)

        self.model = MLPController(input_dim=len(FEATURE_KEYS)).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def decide(self, question, retrieved_summaries, retrieved_chunks, summary_answer, raw_answer):
        features = extract_features(
            question, retrieved_summaries, retrieved_chunks, summary_answer, raw_answer
        )

        x = torch.tensor([features], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            pred = torch.argmax(logits, dim=1).item()

        if pred == 1:
            return {"decision": "raw", "reason": "mlp_controller_lme"}
        return {"decision": "summary", "reason": "mlp_controller_lme"}