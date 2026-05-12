import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

DATA_PATH = Path("data/processed/lme_controller_dataset_mlp.json")
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


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def main():
    rows = load_json(DATA_PATH)

    X = []
    y = []

    for row in rows:
        X.append([float(row[k]) for k in FEATURE_KEYS])
        y.append(int(row["label"]))

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training on device:", device)

    model = MLPController(input_dim=len(FEATURE_KEYS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = 0.0
    best_state = None

    for epoch in range(40):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
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
        "feature_keys": FEATURE_KEYS,
    }, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Best val accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()