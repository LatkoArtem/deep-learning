from __future__ import annotations

import argparse
import csv
import ctypes
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from ctypes import wintypes

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab3"))
from experiment import CLASSES, IMG_SIZE, MEAN, STD, image_path, read_split  # noqa: E402

BATCH_SIZE = 64
CALIBRATION_SIZE = 200
ROUNDS = 3
WARMUPS = 20
MEASUREMENTS = 100


def model_from_checkpoint(path: Path) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("classes") != CLASSES or checkpoint.get("img_size") != IMG_SIZE
            or checkpoint.get("regime") != "frozen"):
        raise ValueError("Checkpoint не відповідає класам або розміру з ЛР3.")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_rows(data_root: Path, split_path: Path) -> tuple[list[dict], list[dict]]:
    rows = read_split(split_path)
    train = sorted((row for row in rows if row["part"] == "train"), key=lambda r: r["file"])
    test = [row for row in rows if row["part"] == "test"]
    if (len(train), len(test)) != (1200, 1502):
        raise ValueError(f"Неправильні розміри поділу: train={len(train)}, test={len(test)}")
    for row in rows:
        if not image_path(data_root, row).is_file():
            raise FileNotFoundError(image_path(data_root, row))
    return train, test


def tensor_for(row: dict, data_root: Path, transform: transforms.Compose) -> torch.Tensor:
    with Image.open(image_path(data_root, row)) as source:
        return transform(source.convert("RGB"))


def batch_tensors(rows: list[dict], data_root: Path, transform: transforms.Compose):
    for start in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[start:start + BATCH_SIZE]
        images = torch.stack([tensor_for(row, data_root, transform) for row in batch_rows])
        labels = [CLASSES.index(row["label"]) for row in batch_rows]
        yield images, labels


def verify_split_and_calibration(train: list[dict]) -> tuple[list[dict], dict]:
    indices = np.random.default_rng(0).permutation(1200)[:CALIBRATION_SIZE]
    selected = [train[int(i)] for i in indices]
    counts = Counter(row["label"] for row in selected)
    summary = {
        "count": len(selected),
        "filename_sum": sum(int(Path(row["file"]).stem) for row in selected),
        "class_counts": {name: counts[name] for name in CLASSES},
    }
    expected = {
        "count": 200,
        "filename_sum": 1921470,
        "class_counts": {"buildings": 40, "forest": 35, "glacier": 36,
                         "mountain": 33, "sea": 29, "street": 27},
    }
    if summary != expected:
        raise ValueError(f"Калібрувальна вибірка не збіглася: {summary}")
    return selected, summary


class ImageReader(CalibrationDataReader):
    def __init__(self, rows: list[dict], data_root: Path, input_name: str):
        self.rows = rows
        self.data_root = data_root
        self.input_name = input_name
        self.transform = image_transform()
        self.rewind()

    def get_next(self):
        try:
            row = next(self.rows_iter)
        except StopIteration:
            return None
        image = tensor_for(row, self.data_root, self.transform).unsqueeze(0).numpy()
        return {self.input_name: np.ascontiguousarray(image, dtype=np.float32)}

    def rewind(self):
        self.rows_iter = iter(self.rows)


def make_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def evaluate_torch(model: torch.nn.Module, rows: list[dict], data_root: Path) -> np.ndarray:
    outputs = []
    with torch.no_grad():
        for images, _ in batch_tensors(rows, data_root, image_transform()):
            outputs.append(model(images).numpy())
    return np.concatenate(outputs)


def evaluate_onnx(session: ort.InferenceSession, rows: list[dict], data_root: Path) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs = []
    for images, _ in batch_tensors(rows, data_root, image_transform()):
        outputs.append(session.run(["logits"], {input_name: images.numpy()})[0])
    return np.concatenate(outputs)


def quality(logits: np.ndarray, targets: np.ndarray) -> dict:
    predictions = logits.argmax(axis=1)
    return {
        "macro_f1": float(f1_score(targets, predictions, labels=np.arange(6), average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "predictions": predictions,
    }


def per_channel_ranges(model_path: Path) -> dict:
    model = onnx.load(model_path)
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    values = []
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2 or node.input[1] not in initializers:
            continue
        weights = np.asarray(initializers[node.input[1]])
        channel_max = np.max(np.abs(weights).reshape(weights.shape[0], -1), axis=1)
        median = float(np.median(channel_max))
        ratio = float(np.max(channel_max) / median) if median else float("inf")
        values.append({"node": node.name, "r": ratio, "median_channel_max": median})
    if len(values) != 20:
        raise ValueError(f"Очікувалося 20 Conv із доступними вагами, отримано {len(values)}")
    values.sort(key=lambda item: item["r"], reverse=True)
    return {
        "convolution_count": len(values),
        "top3": values[:3],
        "median_r": float(np.median([item["r"] for item in values])),
        "median_channel_cost_bits": float(np.log2(np.median([item["r"] for item in values]))),
    }


def paired_comparison(fp32: dict, int8: dict, targets: np.ndarray) -> dict:
    f_pred, q_pred = fp32["predictions"], int8["predictions"]
    f_ok, q_ok = f_pred == targets, q_pred == targets
    b = int(np.count_nonzero(q_ok & ~f_ok))
    c = int(np.count_nonzero(~q_ok & f_ok))
    n = len(targets)
    delta = float(int8["accuracy"] - fp32["accuracy"])
    se = float(np.sqrt(b + c) / n)
    threshold = float(2 * np.sqrt(b + c))
    single_se = float(np.sqrt(fp32["accuracy"] * (1 - fp32["accuracy"]) / n))
    return {
        "b_int8_correct_fp32_wrong": b,
        "c_int8_wrong_fp32_correct": c,
        "delta_accuracy": delta,
        "paired_se": se,
        "two_se_count_threshold": threshold,
        "noise_rule_passed": bool(abs(b - c) <= threshold),
        "single_model_se": single_se,
    }


def measure_round(session: ort.InferenceSession, image: np.ndarray) -> list[float]:
    input_name = session.get_inputs()[0].name
    feed = {input_name: image}
    for _ in range(WARMUPS):
        session.run(["logits"], feed)
    values = []
    for _ in range(MEASUREMENTS):
        start = time.perf_counter()
        session.run(["logits"], feed)
        values.append((time.perf_counter() - start) * 1000)
    return values


def summarize_latency(rounds: list[list[float]]) -> dict:
    measurements = [value for round_values in rounds for value in round_values]
    return {
        "p50_ms": float(np.percentile(measurements, 50)),
        "p95_ms": float(np.percentile(measurements, 95)),
        "round_p50_ms": [float(np.percentile(values, 50)) for values in rounds],
        "round_p95_ms": [float(np.percentile(values, 95)) for values in rounds],
        "measurement_count": len(measurements),
        "warmup_count": WARMUPS,
    }


def benchmark_latency(
    fp32_session: ort.InferenceSession,
    int8_session: ort.InferenceSession,
    image: np.ndarray,
) -> tuple[dict, dict]:
    fp32_rounds, int8_rounds = [], []
    for _ in range(ROUNDS):
        fp32_rounds.append(measure_round(fp32_session, image))
        int8_rounds.append(measure_round(int8_session, image))
    return summarize_latency(fp32_rounds), summarize_latency(int8_rounds)


def power_line_status() -> str:
    if sys.platform != "win32":
        return "not_available"

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ac_line_status", wintypes.BYTE),
            ("battery_flag", wintypes.BYTE),
            ("battery_life_percent", wintypes.BYTE),
            ("system_status_flag", wintypes.BYTE),
            ("battery_life_time", wintypes.DWORD),
            ("battery_full_life_time", wintypes.DWORD),
        ]

    status = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return "unknown"
    return {0: "battery", 1: "ac"}.get(status.ac_line_status, "unknown")


def cpu_model() -> str:
    if sys.platform == "win32":
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    return platform.processor()


def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(1)
    power = power_line_status()
    if power == "battery" and not args.allow_battery:
        raise RuntimeError("Підключіть ноутбук до живлення перед запуском замірів затримки.")
    if power == "unknown":
        raise RuntimeError("Не вдалося перевірити живлення. Переконайтеся, що ноутбук підключений до зарядного пристрою.")
    data_root, split_path, checkpoint_path, out = map(
        Path, (args.data, args.split, args.model, args.out)
    )
    out.mkdir(parents=True, exist_ok=True)
    train_rows, test_rows = load_rows(data_root, split_path)
    calibration_rows, calibration_summary = verify_split_and_calibration(train_rows)
    model = model_from_checkpoint(checkpoint_path)

    fp32_path = out / "lab4_fp32.onnx"
    int8_path = out / "lab4_int8.onnx"
    per_tensor_path = out / "lab4_int8_per_tensor.onnx"
    prep_path = out / "lab4_preprocessed.onnx"
    example = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE), dtype=torch.float32)
    print("Експорт FP32 у ONNX...", flush=True)
    start = time.perf_counter()
    torch.onnx.export(
        model, example, str(fp32_path), dynamo=False, opset_version=17,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    export_seconds = time.perf_counter() - start
    graph = onnx.load(fp32_path)
    onnx.checker.check_model(graph)
    graph_counts = Counter(node.op_type for node in graph.graph.node)

    print("Калібрування та квантизація...", flush=True)
    start = time.perf_counter()
    quant_pre_process(str(fp32_path), str(prep_path), skip_symbolic_shape=True)
    reader = ImageReader(calibration_rows, data_root, "input")
    quantize_static(
        str(prep_path), str(int8_path), reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    quantization_seconds = time.perf_counter() - start

    per_tensor_start = time.perf_counter()
    quantize_static(
        str(prep_path), str(per_tensor_path),
        ImageReader(calibration_rows, data_root, "input"),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
        calibrate_method=CalibrationMethod.MinMax,
    )
    per_tensor_seconds = time.perf_counter() - per_tensor_start
    prep_path.unlink(missing_ok=True)

    print("Обчислення якості на test...", flush=True)
    fp32_session = make_session(fp32_path)
    int8_session = make_session(int8_path)
    per_tensor_session = make_session(per_tensor_path)
    labels = np.array([CLASSES.index(row["label"]) for row in test_rows], dtype=np.int64)
    torch_logits = evaluate_torch(model, test_rows, data_root)
    fp32_logits = evaluate_onnx(fp32_session, test_rows, data_root)
    int8_logits = evaluate_onnx(int8_session, test_rows, data_root)
    per_tensor_logits = evaluate_onnx(per_tensor_session, test_rows, data_root)

    torch_quality = quality(torch_logits, labels)
    fp32_quality = quality(fp32_logits, labels)
    int8_quality = quality(int8_logits, labels)
    per_tensor_quality = quality(per_tensor_logits, labels)
    agreement = float(np.mean(torch_logits.argmax(1) == fp32_logits.argmax(1)))
    max_logit_diff = float(np.max(np.abs(torch_logits - fp32_logits)))
    if max_logit_diff >= 1e-3 or agreement != 1.0:
        raise ValueError(f"Експорт не пройшов контроль: max_diff={max_logit_diff}, agreement={agreement}")

    paired = paired_comparison(fp32_quality, int8_quality, labels)
    per_tensor_paired = paired_comparison(fp32_quality, per_tensor_quality, labels)
    channel_stats = per_channel_ranges(fp32_path)
    sample = tensor_for(test_rows[0], data_root, image_transform()).unsqueeze(0).numpy()
    print("Вимірювання затримки на CPU...", flush=True)
    latency_fp32, latency_int8 = benchmark_latency(fp32_session, int8_session, sample)
    latency_ranges = {
        "fp32": [min(latency_fp32["round_p50_ms"]), max(latency_fp32["round_p50_ms"])],
        "int8": [min(latency_int8["round_p50_ms"]), max(latency_int8["round_p50_ms"])],
    }
    latency_changed = bool(
        latency_ranges["fp32"][1] < latency_ranges["int8"][0]
        or latency_ranges["int8"][1] < latency_ranges["fp32"][0]
    )

    def file_mb(path: Path) -> float:
        return path.stat().st_size / 1_000_000

    result = {
        "environment": {
            "os": platform.platform(),
            "cpu": cpu_model(),
            "cpu_vector_capability": torch.backends.cpu.get_cpu_capability(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "threads": 1,
            "provider": "CPUExecutionProvider",
            "batch_size": 1,
            "power_line": power,
        },
        "data": {
            "train_count": len(train_rows),
            "train_filename_sum": sum(int(Path(row["file"]).stem) for row in train_rows),
            "test_count": len(test_rows),
            "test_filename_sum": sum(int(Path(row["file"]).stem) for row in test_rows),
            "calibration": calibration_summary,
        },
        "export": {
            "max_abs_logit_difference": max_logit_diff,
            "class_agreement": agreement,
            "pytorch_macro_f1": torch_quality["macro_f1"],
            "onnx_fp32_macro_f1": fp32_quality["macro_f1"],
            "conv_nodes": graph_counts["Conv"],
            "batch_normalization_nodes": graph_counts["BatchNormalization"],
            "seconds": export_seconds,
        },
        "quantization": {"seconds": quantization_seconds},
        "quality": {
            "pytorch_fp32": {k: v for k, v in torch_quality.items() if k != "predictions"},
            "onnx_fp32": {k: v for k, v in fp32_quality.items() if k != "predictions"},
            "int8_per_channel": {
                **{k: v for k, v in int8_quality.items() if k != "predictions"},
                "agreement_with_fp32": float(np.mean(int8_quality["predictions"] == fp32_quality["predictions"])),
                **paired,
            },
            "int8_per_tensor": {
                **{k: v for k, v in per_tensor_quality.items() if k != "predictions"},
                "agreement_with_fp32": float(np.mean(per_tensor_quality["predictions"] == fp32_quality["predictions"])),
                **per_tensor_paired,
            },
        },
        "weight_ranges": channel_stats,
        "per_tensor_experiment_seconds": per_tensor_seconds,
        "latency": {
            "fp32": {"file_mb": file_mb(fp32_path), **latency_fp32},
            "int8": {"file_mb": file_mb(int8_path), **latency_int8},
            "speedup_p50": latency_fp32["p50_ms"] / latency_int8["p50_ms"],
            "round_p50_ranges_ms": latency_ranges,
            "change_measurable": latency_changed,
        },
    }
    result_path = out / "lab4_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Результати збережено: {result_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Експорт і порівняння FP32 та INT8 для ЛР4")
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint lab3_model.pt з ЛР3")
    parser.add_argument("--split", type=Path, required=True, help="split.csv з ЛР3")
    parser.add_argument("--data", type=Path, required=True, help="Локальна тека Intel Image Classification")
    parser.add_argument("--out", type=Path, default=Path("."), help="Тека для локальних результатів")
    parser.add_argument("--allow-battery", action="store_true", help="Дозволити заміри без живлення та записати відхилення")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
