"""ЛР2. Підготовка даних, навчання та збереження прогнозів."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 0
EPOCHS = 12
BATCH = 1024
LR = 1e-3
HIDDEN = 32
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    amount: np.ndarray
    row_ids: np.ndarray


def make_splits(csv: Path) -> tuple[Split, Split, Split, StandardScaler]:
    df = pd.read_csv(csv)
    missing = sorted(set(FEATURES + ["Class"]) - set(df.columns))
    if missing:
        raise ValueError(f"У CSV бракує стовпців: {', '.join(missing)}")
    if not df["Class"].isin([0, 1]).all():
        raise ValueError("Class має містити лише мітки 0/1.")
    y = df["Class"].to_numpy(dtype=np.int64)
    amount = df["Amount"].to_numpy(dtype=np.float64)
    x = df[FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Ознаки містять пропущені або нескінченні значення.")
    idx = np.arange(len(y))
    idx_tr, idx_tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
    idx_va, idx_te = train_test_split(
        idx_tmp, test_size=0.5, stratify=y[idx_tmp], random_state=SEED
    )
    scaler = StandardScaler().fit(x[idx_tr])
    tr, va, te = (
        Split(scaler.transform(x[i]).astype(np.float32), y[i], amount[i], i)
        for i in (idx_tr, idx_va, idx_te)
    )
    return tr, va, te, scaler


def train_once(s: Split) -> torch.nn.Module:
    torch.manual_seed(SEED)
    model = torch.nn.Sequential(
        torch.nn.Linear(len(FEATURES), HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1),
    ).to(device="cpu", dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt = torch.from_numpy(s.x)
    yt = torch.from_numpy(s.y.astype(np.float32)).unsqueeze(1)
    rng = np.random.default_rng(SEED)
    model.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(s.y))
        total = 0.0
        for start in range(0, len(s.y), BATCH):
            batch = torch.from_numpy(order[start : start + BATCH])
            opt.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(batch)
        print(f"  епоха {epoch + 1}/{EPOCHS}  BCE={total / len(s.y):.6f}")
    model.eval()
    return model


@torch.no_grad()
def scores(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(x))).squeeze(1).numpy()


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_run(
    out: Path,
    model: torch.nn.Module,
    splits: tuple[Split, Split, Split],
    scaler: StandardScaler,
    source: Path,
) -> dict[str, np.ndarray]:
    saved = {}
    for name, split in zip(("train", "validation", "test"), splits, strict=True):
        saved[f"{name}_row_ids"] = split.row_ids
        saved[f"{name}_y"] = split.y
        saved[f"{name}_amount"] = split.amount
        if name != "train":
            saved[f"{name}_scores"] = scores(model, split.x)
    torch.save(model.state_dict(), out / "model.pt")
    np.savez_compressed(out / "predictions.npz", **saved)
    np.savez_compressed(
        out / "preprocessing.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
        n_samples_seen=scaler.n_samples_seen_,
        features=np.asarray(FEATURES),
    )
    write_json(
        out / "run.json",
        {
            "source_csv": str(source.resolve()),
            "features": FEATURES,
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "learning_rate": LR,
            "architecture": [len(FEATURES), HIDDEN, 1],
            "dtype": "float32",
            "device": "cpu",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": str(torch.__version__),
            "torch_num_threads": torch.get_num_threads(),
        },
    )
    return saved


def load_predictions(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def split_summary(saved: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name in ("train", "validation", "test"):
        y = saved[f"{name}_y"]
        positives = int(np.count_nonzero(y == 1))
        rows.append(
            {
                "split": name,
                "n": len(y),
                "positives": positives,
                "positive_fraction": positives / len(y),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--data", type=Path)
    mode.add_argument("--saved", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.saved is not None:
        if args.out is not None:
            parser.error("У режимі --saved параметр --out не потрібний.")
        saved = load_predictions(args.saved)
        print(split_summary(saved).to_string(index=False))
        print("Прогнози завантажено. Запустіть analyze_thresholds.py для аналізу.")
        return
    if args.out is None or args.data is None:
        parser.error("Для навчання вкажіть --data і --out.")
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.error("Тека --out непорожня або шлях зайнятий файлом.")
    train, validation, test, scaler = make_splits(args.data)
    args.out.mkdir(parents=True, exist_ok=True)
    model = train_once(train)
    saved = save_run(args.out, model, (train, validation, test), scaler, args.data)
    summary = split_summary(saved)
    summary.to_csv(args.out / "split_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Ваги й прогнози збережено: {args.out.resolve()}")


if __name__ == "__main__":
    main()

