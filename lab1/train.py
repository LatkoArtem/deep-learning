from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 0
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
EPOCHS = 12
BATCH = 1024


def load_data(path: Path):
    df = pd.read_csv(path)
    missing = sorted(set(FEATURES + ["Class"]) - set(df.columns))
    if missing:
        raise ValueError(f"У CSV бракує стовпців: {', '.join(missing)}")

    y = df["Class"].to_numpy(dtype=np.int64)
    x = df[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Перевірте ознаки та значення Class у CSV.")

    ids = np.arange(len(y))
    train_ids, other_ids = train_test_split(
        ids, test_size=0.4, stratify=y, random_state=SEED
    )
    validation_ids, test_ids = train_test_split(
        other_ids, test_size=0.5, stratify=y[other_ids], random_state=SEED
    )

    scaler = StandardScaler().fit(x[train_ids])
    return (
        scaler.transform(x[train_ids]).astype(np.float32),
        y[train_ids],
        scaler.transform(x[validation_ids]).astype(np.float32),
        y[validation_ids],
        scaler.transform(x[test_ids]).astype(np.float32),
        y[test_ids],
    )


def train_model(x: np.ndarray, y: np.ndarray) -> torch.nn.Module:
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURES), 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    inputs = torch.from_numpy(x)
    targets = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    rng = np.random.default_rng(SEED)

    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(y))
        total_loss = 0.0
        for start in range(0, len(y), BATCH):
            batch = torch.from_numpy(order[start : start + BATCH])
            optimizer.zero_grad()
            loss = loss_fn(model(inputs[batch]), targets[batch])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f"епоха {epoch + 1}/{EPOCHS}  BCE={total_loss / len(y):.6f}")
    return model.eval()


@torch.no_grad()
def predict(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    return torch.sigmoid(model(torch.from_numpy(x))).squeeze(1).numpy()


def metrics(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    predicted = (scores >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Тека --out непорожня.")

    train_x, train_y, validation_x, validation_y, test_x, test_y = load_data(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    model = train_model(train_x, train_y)
    validation_scores = predict(model, validation_x)
    test_scores = predict(model, test_x)

    np.savez_compressed(
        args.out / "predictions.npz",
        validation_y=validation_y,
        validation_scores=validation_scores,
        test_y=test_y,
        test_scores=test_scores,
    )
    result = {
        "validation": metrics(validation_y, validation_scores),
        "test": metrics(test_y, test_scores),
    }
    (args.out / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

