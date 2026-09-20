from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_iris

EPS = 1e-6
TORCH_TOLERANCE = 1e-12
NUMERICAL_TOLERANCE = 1e-7


@dataclass
class Parameters:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


@dataclass
class Cache:
    x: np.ndarray
    y: np.ndarray
    z1: np.ndarray
    a1: np.ndarray
    z2: np.ndarray
    log_probs: np.ndarray


def split_iris() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    iris = load_iris()
    rng = np.random.default_rng(0)
    train_ids = []
    test_ids = []
    for label in range(3):
        ids = np.flatnonzero(iris.target == label).copy()
        rng.shuffle(ids)
        train_ids.extend(ids[:35])
        test_ids.extend(ids[35:])

    train_ids = np.asarray(train_ids, dtype=np.int64)
    test_ids = np.asarray(test_ids, dtype=np.int64)
    mean = iris.data[train_ids].mean(axis=0)
    std = iris.data[train_ids].std(axis=0, ddof=0)
    x_train = (iris.data[train_ids] - mean) / std
    x_test = (iris.data[test_ids] - mean) / std
    return x_train.astype(np.float64), iris.target[train_ids], x_test.astype(np.float64), iris.target[test_ids]


def initial_parameters() -> Parameters:
    rng = np.random.default_rng(0)
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4.0), size=(4, 8))
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8.0 + 3.0)), size=(8, 3))
    return Parameters(
        W1=W1.astype(np.float64),
        b1=np.zeros(8, dtype=np.float64),
        W2=W2.astype(np.float64),
        b2=np.zeros(3, dtype=np.float64),
    )


def forward(parameters: Parameters, x: np.ndarray, y: np.ndarray) -> tuple[float, Cache]:
    z1 = x @ parameters.W1 + parameters.b1
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ parameters.W2 + parameters.b2
    shifted = z2 - z2.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    loss = float(-log_probs[np.arange(len(y)), y].mean())
    return loss, Cache(x=x, y=y, z1=z1, a1=a1, z2=z2, log_probs=log_probs)


def backward(parameters: Parameters, cache: Cache, wrong_gradient: bool = False) -> Parameters:
    dz2 = np.exp(cache.log_probs)
    dz2[np.arange(len(cache.y)), cache.y] -= 1.0
    if not wrong_gradient:
        dz2 /= len(cache.y)
    dW2 = cache.a1.T @ dz2
    db2 = dz2.sum(axis=0)
    dz1 = (dz2 @ parameters.W2.T) * (cache.z1 > 0.0)
    dW1 = cache.x.T @ dz1
    db1 = dz1.sum(axis=0)
    return Parameters(dW1, db1, dW2, db2)


def torch_reference(parameters: Parameters, x: np.ndarray, y: np.ndarray) -> tuple[float, Parameters]:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8, dtype=torch.float64),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 3, dtype=torch.float64),
    )
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(parameters.W1.T))
        model[0].bias.copy_(torch.from_numpy(parameters.b1))
        model[2].weight.copy_(torch.from_numpy(parameters.W2.T))
        model[2].bias.copy_(torch.from_numpy(parameters.b2))

    inputs = torch.from_numpy(x)
    targets = torch.from_numpy(y)
    loss = torch.nn.functional.cross_entropy(model(inputs), targets)
    loss.backward()
    gradients = Parameters(
        model[0].weight.grad.detach().numpy().T.copy(),
        model[0].bias.grad.detach().numpy().copy(),
        model[2].weight.grad.detach().numpy().T.copy(),
        model[2].bias.grad.detach().numpy().copy(),
    )
    return float(loss.detach().numpy()), gradients


def max_differences(left: Parameters, right: Parameters) -> dict[str, float]:
    return {
        name: float(np.max(np.abs(getattr(left, name) - getattr(right, name))))
        for name in ("W1", "b1", "W2", "b2")
    }


def numerical_derivative(
    parameters: Parameters,
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    index: tuple[int, ...],
) -> float:
    value = getattr(parameters, name)
    original = float(value[index])
    value[index] = original + EPS
    loss_plus = forward(parameters, x, y)[0]
    value[index] = original - EPS
    loss_minus = forward(parameters, x, y)[0]
    value[index] = original
    return (loss_plus - loss_minus) / (2.0 * EPS)


def numerical_checks(
    parameters: Parameters,
    x: np.ndarray,
    y: np.ndarray,
    gradients: Parameters,
) -> list[dict[str, object]]:
    selected = (("W1", (0, 0)), ("b1", (0,)), ("W2", (0, 0)), ("b2", (0,)))
    rows = []
    for name, index in selected:
        numerical = numerical_derivative(parameters, x, y, name, index)
        manual = float(getattr(gradients, name)[index])
        difference = abs(manual - numerical)
        rows.append(
            {
                "parameter": f"{name}{index}",
                "manual": manual,
                "numerical": numerical,
                "absolute_difference": difference,
                "passed": bool(difference <= NUMERICAL_TOLERANCE),
            }
        )
    return rows


def run_checks(wrong_gradient: bool) -> dict[str, object]:
    x_train, y_train, x_test, y_test = split_iris()
    parameters = initial_parameters()
    numpy_loss, cache = forward(parameters, x_train, y_train)
    numpy_gradients = backward(parameters, cache, wrong_gradient=wrong_gradient)
    torch_loss, torch_gradients = torch_reference(parameters, x_train, y_train)
    differences = max_differences(numpy_gradients, torch_gradients)
    numerical = numerical_checks(parameters, x_train, y_train, numpy_gradients)

    correct_torch_check = abs(numpy_loss - torch_loss) <= TORCH_TOLERANCE and all(
        value <= TORCH_TOLERANCE for value in differences.values()
    )
    numerical_check = all(row["passed"] for row in numerical)
    result = {
        "wrong_gradient": wrong_gradient,
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "train_class_counts": np.bincount(y_train, minlength=3).tolist(),
        "test_class_counts": np.bincount(y_test, minlength=3).tolist(),
        "numpy_loss": numpy_loss,
        "torch_loss": torch_loss,
        "loss_absolute_difference": abs(numpy_loss - torch_loss),
        "gradient_max_absolute_differences": differences,
        "torch_check_passed": bool(correct_torch_check),
        "numerical_checks": numerical,
        "numerical_check_passed": bool(numerical_check),
        "wrong_gradient_detected": bool(wrong_gradient and (not correct_torch_check or not numerical_check)),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrong-gradient", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or Path("wrong_gradient_results.json" if args.wrong_gradient else "results.json")
    result = run_checks(args.wrong_gradient)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
