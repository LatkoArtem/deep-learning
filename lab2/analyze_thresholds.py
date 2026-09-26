from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from lab2_starter import load_predictions

FN_COST = 10
FP_COST = 1


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
    ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
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


def brute_force_table(y: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    thresholds = [float("inf"), *sorted({float(value) for value in scores}, reverse=True)]
    return pd.DataFrame([add_cost(counts(y, scores, threshold)) for threshold in thresholds])


def same_threshold(left: float, right: float) -> bool:
    return (np.isinf(left) and np.isinf(right)) or np.isclose(left, right)


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


def toy_sets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {
        "A": (
            np.array([0, 1, 0, 1, 0, 0]),
            np.array([0.9, 0.7, 0.7, 0.4, 0.2, 0.1]),
        ),
        "B": (
            np.array([0, 0, 0]),
            np.array([0.9, 0.6, 0.3]),
        ),
        "C": (
            np.array([1, 1, *([0] * 10), 0]),
            np.array([0.8, *([0.5] * 11), 0.2]),
        ),
    }


def check_toy_sets() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    detail_rows = []
    for name, (y, scores) in toy_sets().items():
        fast = sorted_candidates(y, scores)
        direct = brute_force_table(y, scores)
        counters_match = (
            len(fast) == len(direct)
            and np.array_equal(
                fast[["tp", "fp", "fn", "tn", "cost"]].to_numpy(),
                direct[["tp", "fp", "fn", "tn", "cost"]].to_numpy(),
            )
        )
        fast_best = choose_best(fast)
        direct_best = choose_best(direct)
        selected_match = (
            counters_match
            and same_threshold(float(fast_best["threshold"]), float(direct_best["threshold"]))
        )
        if not selected_match:
            raise AssertionError(f"Перевірка набору {name} не пройдена.")

        summary_rows.append({
            "set": name,
            "selected_threshold": float(fast_best["threshold"]),
            "tp": int(fast_best["tp"]),
            "fp": int(fast_best["fp"]),
            "fn": int(fast_best["fn"]),
            "tn": int(fast_best["tn"]),
            "cost": int(fast_best["cost"]),
            "counters_match": counters_match,
            "selected_match": selected_match,
        })
        for row_fast, row_direct in zip(
            fast.to_dict("records"), direct.to_dict("records"), strict=True
        ):
            detail_rows.append({
                "set": name,
                "threshold": row_fast["threshold"],
                "fast_tp": row_fast["tp"],
                "fast_fp": row_fast["fp"],
                "fast_fn": row_fast["fn"],
                "fast_tn": row_fast["tn"],
                "fast_cost": row_fast["cost"],
                "direct_tp": row_direct["tp"],
                "direct_fp": row_direct["fp"],
                "direct_fn": row_direct["fn"],
                "direct_tn": row_direct["tn"],
                "direct_cost": row_direct["cost"],
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def save_curves(out: Path, y: np.ndarray, scores: np.ndarray) -> float:
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
    return roc_auc


def save_errors(
    out: Path,
    row_ids: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    amount: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predicted = scores >= threshold
    error_type = np.where(
        (y == 0) & predicted, "FP", np.where((y == 1) & ~predicted, "FN", "")
    )
    all_errors = pd.DataFrame({
        "csv_row": row_ids,
        "error": error_type,
        "label": y,
        "score": scores,
        "amount": amount,
        "distance_from_threshold": np.abs(scores - threshold),
    })
    all_errors = all_errors[all_errors["error"] != ""].sort_values("csv_row")
    selected = pd.concat(
        [all_errors[all_errors["error"] == name].head(5) for name in ("FP", "FN")],
        ignore_index=True,
    ).sort_values(["error", "csv_row"])
    selected.to_csv(out / "test_errors.csv", index=False)
    return all_errors, selected


def safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved", type=Path, required=True)
    args = parser.parse_args()
    out = args.saved
    data = load_predictions(out)
    y_validation = data["validation_y"]
    s_validation = data["validation_scores"]

    candidates = sorted_candidates(y_validation, s_validation)
    candidates.rename(columns={
        "tp": "TP", "fp": "FP", "fn": "FN", "tn": "TN", "cost": "C"
    }).to_csv(out / "validation_threshold_candidates.csv", index=False)
    best = choose_best(candidates)
    threshold = float(best["threshold"])

    comparison = pd.DataFrame([
        {"rule": "standard", **add_cost(counts(y_validation, s_validation, 0.5))},
        {"rule": "minimum_cost", **best.to_dict()},
        {"rule": "all_negative", **add_cost(counts(y_validation, s_validation, float("inf")))},
    ])
    comparison.rename(columns={
        "tp": "TP", "fp": "FP", "fn": "FN", "tn": "TN", "cost": "C"
    }).to_csv(out / "validation_threshold_comparison.csv", index=False)

    toy_summary, toy_details = check_toy_sets()
    toy_summary.to_csv(out / "toy_sets_summary.csv", index=False)
    toy_details.to_csv(out / "toy_sets_check.csv", index=False)
    roc_auc = save_curves(out, y_validation, s_validation)

    test_counts = add_cost(counts(data["test_y"], data["test_scores"], threshold))
    all_errors, selected_errors = save_errors(
        out,
        data["test_row_ids"],
        data["test_y"],
        data["test_scores"],
        data["test_amount"],
        threshold,
    )
    report = {
        "validation": {
            "roc_auc": roc_auc,
            "selected_threshold": "+inf" if np.isinf(threshold) else threshold,
            "selected": json_row(best),
        },
        "test": {
            **json_row(test_counts),
            "precision": safe_rate(int(test_counts["tp"]), int(test_counts["tp"] + test_counts["fp"])),
            "recall": safe_rate(int(test_counts["tp"]), int(test_counts["tp"] + test_counts["fn"])),
        },
        "test_error_rows_total": int(len(all_errors)),
        "test_error_rows_saved": int(len(selected_errors)),
        "toy_checks_passed": bool(toy_summary["selected_match"].all()),
    }
    (out / "threshold_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
