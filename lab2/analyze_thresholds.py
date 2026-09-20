from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions

FP_COST = 1
FN_COST = 10


def counts(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int | float]:
    predicted = scores >= threshold
    return {
        "threshold": float(threshold),
        "tp": int(np.count_nonzero((y == 1) & predicted)),
        "fp": int(np.count_nonzero((y == 0) & predicted)),
        "fn": int(np.count_nonzero((y == 1) & ~predicted)),
        "tn": int(np.count_nonzero((y == 0) & ~predicted)),
    }


def add_cost(row: dict[str, int | float]) -> dict[str, int | float]:
    row["cost"] = FN_COST * int(row["fn"]) + FP_COST * int(row["fp"])
    return row


def sorted_candidates(y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_y = y[order]
    ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    tp = np.cumsum(sorted_y == 1)[ends]
    fp = np.cumsum(sorted_y == 0)[ends]
    positives = int(np.count_nonzero(y == 1))
    negatives = int(np.count_nonzero(y == 0))

    rows = [add_cost({
        "threshold": float("inf"),
        "tp": 0,
        "fp": 0,
        "fn": positives,
        "tn": negatives,
    })]
    for threshold, current_tp, current_fp in zip(
        sorted_scores[ends], tp, fp, strict=True
    ):
        rows.append(add_cost({
            "threshold": float(threshold),
            "tp": int(current_tp),
            "fp": int(current_fp),
            "fn": positives - int(current_tp),
            "tn": negatives - int(current_fp),
        }))
    return pd.DataFrame(rows)


def choose_best(table: pd.DataFrame) -> pd.Series:
    return table.sort_values(
        ["cost", "threshold"], ascending=[True, False], kind="stable"
    ).iloc[0]


def brute_force(y: np.ndarray, scores: np.ndarray) -> dict[str, int | float]:
    thresholds = [float("inf"), *sorted({float(value) for value in scores}, reverse=True)]
    rows = [add_cost(counts(y, scores, threshold)) for threshold in thresholds]
    return min(rows, key=lambda row: (int(row["cost"]), -float(row["threshold"])))


def same_threshold(left: float, right: float) -> bool:
    return np.isinf(left) and np.isinf(right) or np.isclose(left, right)


def json_row(row: pd.Series | dict) -> dict:
    values = row.to_dict() if isinstance(row, pd.Series) else row
    result = {}
    for key, value in values.items():
        if key == "threshold" and np.isinf(float(value)):
            result[key] = "+inf"
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def check_toy_sets() -> pd.DataFrame:
    toy_sets = {
        "A": (np.array([0, 1, 0, 1]), np.array([0.10, 0.40, 0.70, 0.90])),
        "B": (np.array([1, 0, 0, 1]), np.array([0.20, 0.30, 0.60, 0.80])),
        "C": (np.array([0, 1, 0, 1, 0]), np.array([0.30, 0.30, 0.60, 0.60, 0.90])),
    }
    rows = []
    for name, (y, scores) in toy_sets.items():
        fast = choose_best(sorted_candidates(y, scores))
        direct = brute_force(y, scores)
        ok = (
            int(fast["cost"]) == int(direct["cost"])
            and same_threshold(float(fast["threshold"]), float(direct["threshold"]))
        )
        if not ok:
            raise AssertionError(f"Перевірка набору {name} не пройдена.")
        rows.append({"set": name, "threshold": float(fast["threshold"]), "cost": int(fast["cost"]), "ok": ok})
    return pd.DataFrame(rows)


def save_curves(out: Path, y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, _ = precision_recall_curve(y, scores)
    fpr, tpr, _ = roc_curve(y, scores)
    roc_auc = float(roc_auc_score(y, scores))

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Validation PR curve")
    plt.tight_layout()
    plt.savefig(out / "validation_pr.png", dpi=140)
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Validation ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "validation_roc.png", dpi=140)
    plt.close()
    return {"roc_auc": roc_auc}


def save_errors(
    out: Path,
    split: str,
    row_ids: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    amount: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    predicted = scores >= threshold
    error_type = np.where((y == 0) & predicted, "FP", np.where((y == 1) & ~predicted, "FN", ""))
    errors = pd.DataFrame({
        "row_id": row_ids,
        "amount": amount,
        "y": y,
        "score": scores,
        "error": error_type,
    })
    errors = errors[errors["error"] != ""].copy()
    errors = errors.sort_values(["error", "score"], ascending=[True, False])
    errors.to_csv(out / f"{split}_errors.csv", index=False)
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved", type=Path, required=True)
    args = parser.parse_args()
    out = args.saved
    data = load_predictions(out)
    y_validation = data["validation_y"]
    s_validation = data["validation_scores"]
    candidates = sorted_candidates(y_validation, s_validation)
    candidates.to_csv(out / "validation_threshold_candidates.csv", index=False)
    best = choose_best(candidates)
    threshold = float(best["threshold"])

    comparison = pd.DataFrame([
        {"name": "0.5", **add_cost(counts(y_validation, s_validation, 0.5))},
        {"name": "t_star", **best.to_dict()},
        {"name": "+inf", **add_cost(counts(y_validation, s_validation, float("inf")))},
    ])
    comparison.to_csv(out / "validation_threshold_comparison.csv", index=False)
    toy_results = check_toy_sets()
    toy_results.to_csv(out / "toy_sets_check.csv", index=False)
    curve_metrics = save_curves(out, y_validation, s_validation)

    test_counts = add_cost(counts(
        data["test_y"], data["test_scores"], threshold
    ))
    errors = save_errors(
        out,
        "test",
        data["test_row_ids"],
        data["test_y"],
        data["test_scores"],
        data["test_amount"],
        threshold,
    )
    report = {
        "validation": {
            "roc_auc": curve_metrics["roc_auc"],
            "selected_threshold": threshold,
            "selected": json_row(best),
        },
        "test": json_row(test_counts),
        "test_error_rows": int(len(errors)),
    }
    if np.isinf(report["validation"]["selected_threshold"]):
        report["validation"]["selected_threshold"] = "+inf"
    (out / "threshold_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
