import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


DATA_PATH = Path("data/processed/controller_dataset_wd.json")
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


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def main():
    rows = load_json(DATA_PATH)

    X_wide = []
    X_deep = []
    y = []

    for row in rows:
        X_wide.append([float(row[k]) for k in WIDE_KEYS])
        X_deep.append([float(row[k]) for k in DEEP_KEYS])
        y.append(int(row["label"]))

    X_wide = torch.tensor(X_wide, dtype=torch.float32)
    X_deep = torch.tensor(X_deep, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_wide, X_deep, y)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on device:", device)

    model = WideDeepController(wide_dim=len(WIDE_KEYS), deep_dim=len(DEEP_KEYS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = 0.0
    best_state = None

    for epoch in range(30):
        model.train()
        total_loss = 0.0

        for wide_x, deep_x, yb in train_loader:
            wide_x = wide_x.to(device)
            deep_x = deep_x.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(wide_x, deep_x)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for wide_x, deep_x, yb in val_loader:
                wide_x = wide_x.to(device)
                deep_x = deep_x.to(device)
                yb = yb.to(device)

                logits = model(wide_x, deep_x)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = correct / total if total > 0 else 0.0

        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict()

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss:.4f} | Val Acc: {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "wide_keys": WIDE_KEYS,
        "deep_keys": DEEP_KEYS,
    }, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Best val accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()