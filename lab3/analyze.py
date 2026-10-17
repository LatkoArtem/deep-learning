from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18

from experiment import (
    CLASSES, IMG_SIZE, REGIMES, RUNS, choose_run, image_path, loader, logits,
    macro_f1, read_split, split_hash, state_hash, write_csv, write_json,
)


def load_journals(output: Path, rows: list[dict]) -> list[dict]:
    current_hash = split_hash(output)
    expected_names = {f'{regime}_seed{seed}' for regime, seed in RUNS}
    present_names = {p.parent.name for p in (output / 'runs').glob('*/run.json')}
    if present_names != expected_names:
        raise ValueError('Потрібні рівно сім журналів за протоколом.')
    labels = np.array([CLASSES.index(row['label']) for row in rows if row['part'] == 'validation'])
    journals = []
    for regime, seed in RUNS:
        folder = output / 'runs' / f'{regime}_seed{seed}'
        journal = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        if journal['split_sha256'] != current_hash:
            raise ValueError(f'Хеш split.csv не збігається: {folder.name}')
        if (journal['regime'], journal['seed']) != (regime, seed):
            raise ValueError(f'Не збігається режим/зерно: {folder.name}')
        history = journal['history']
        if [epoch['epoch'] for epoch in history] != list(range(1, 9)):
            raise ValueError('Потрібні результати всіх восьми епох.')
        if journal['settings']['epochs'] != 8 or journal['settings']['optimizer_steps'] != 304:
            raise ValueError('Порушено бюджет навчання.')
        values = np.load(folder / 'val_logits.npy', allow_pickle=False)
        if values.shape != (1498, 6) or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f'Некоректні логіти: {folder.name}')
        if abs(macro_f1(labels, values) - history[-1]['validation_macro_f1']) > 1e-12:
            raise ValueError(f'Логіти не відповідають журналу: {folder.name}')
        journals.append(journal)
    return journals


def effects(journals: list[dict]) -> dict:
    scores = {r: [j['history'][-1]['validation_macro_f1'] for j in journals if j['regime'] == r] for r in REGIMES}
    mean_frozen = float(np.mean(scores['frozen']))
    mean_unfreeze = float(np.mean(scores['unfreeze']))
    range_frozen = float(np.ptp(scores['frozen']))
    range_unfreeze = float(np.ptp(scores['unfreeze']))
    noise = max(range_frozen, range_unfreeze)
    pre = mean_frozen - scores['scratch'][0]
    unf = mean_unfreeze - mean_frozen
    return {'mean_frozen': mean_frozen, 'range_frozen': range_frozen,
            'mean_unfreeze': mean_unfreeze, 'range_unfreeze': range_unfreeze, 'R': noise,
            'delta_pre': pre, 'delta_unf': unf, 'total_gain': pre + unf,
            'pre_exceeds_noise': abs(pre) > noise, 'unfreeze_exceeds_noise': abs(unf) > noise}


def training_figure(journals: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {'scratch': 'tab:red', 'frozen': 'tab:blue', 'unfreeze': 'tab:green'}
    styles = {1: '-', 2: '--', 3: ':'}
    for journal in journals:
        name = f"{journal['regime']} seed {journal['seed']}"
        x = [row['epoch'] for row in journal['history']]
        for ax, key in zip(axes, ('train_loss', 'validation_macro_f1'), strict=True):
            ax.plot(x, [row[key] for row in journal['history']], color=colors[journal['regime']],
                    linestyle=styles[journal['seed']], label=name)
    axes[0].set_ylabel('Train cross-entropy')
    axes[1].set_ylabel('Validation macro-F1')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.set_xticks(range(1, 9))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def confusion_figure(matrix: np.ndarray, score: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(6), CLASSES, rotation=35, ha='right')
    ax.set_yticks(range(6), CLASSES)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title(f'Test macro-F1 = {score:.4f}')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center',
                    color='white' if matrix[i, j] > matrix.max() / 2 else 'black')
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def select_errors(rows: list[dict], target: np.ndarray, values: np.ndarray, matrix: np.ndarray) -> list[dict]:
    predicted = values.argmax(axis=1)
    probabilities = torch.softmax(torch.from_numpy(values), dim=1).numpy()
    pairs = sorted(((i, j) for i in range(6) for j in range(6) if i != j and matrix[i, j] > 0),
                   key=lambda pair: (-int(matrix[pair]), pair[0], pair[1]))
    candidates = {pair: sorted(np.flatnonzero((target == pair[0]) & (predicted == pair[1])),
                               key=lambda index: rows[index]['file']) for pair in pairs}
    result = []
    rank = 0
    while len(result) < min(5, int((target != predicted).sum())):
        for i, j in pairs:
            if rank < len(candidates[(i, j)]):
                index = candidates[(i, j)][rank]
                result.append({'file': rows[index]['file'], 'true_class': CLASSES[i], 'predicted_class': CLASSES[j],
                               'probability': float(probabilities[index, j]), 'pair_count': int(matrix[i, j])})
                if len(result) == 5:
                    break
        rank += 1
    return result


def error_figure(data: Path, errors: list[dict], path: Path) -> None:
    if not errors:
        raise ValueError('Немає помилок для рисунка.')
    fig, axes = plt.subplots(1, len(errors), figsize=(3.3 * len(errors), 3.5), squeeze=False)
    for ax, error in zip(axes.flat, errors, strict=True):
        row = {'file': error['file'], 'label': error['true_class'], 'part': 'test'}
        with Image.open(image_path(data, row)) as source:
            ax.imshow(source.convert('RGB'))
        ax.set_title(error['file'])
        ax.set_xlabel(f"{error['true_class']} -> {error['predicted_class']}\np={error['probability']:.3f}, pair n={error['pair_count']}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze(data: Path, output: Path) -> dict:
    torch.set_num_threads(6)
    rows = read_split(output / 'split.csv')
    journals = load_journals(output, rows)
    chosen = choose_run(journals)
    checkpoint = torch.load(output / 'lab3_model.pt', map_location='cpu', weights_only=True)
    if (checkpoint['regime'], checkpoint['seed']) != (chosen['regime'], chosen['seed']):
        raise ValueError('Checkpoint не відповідає вибору за validation.')
    if checkpoint['classes'] != CLASSES or checkpoint['img_size'] != IMG_SIZE:
        raise ValueError('Checkpoint має інші класи або розмір зображення.')
    if state_hash(checkpoint['state_dict']) != chosen['state_sha256']:
        raise ValueError('У checkpoint ваги іншого запуску.')
    result_dir = output / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = [{'run': i, 'regime': j['regime'], 'seed': j['seed'], 'train_loss': j['history'][-1]['train_loss'],
                'validation_macro_f1': j['history'][-1]['validation_macro_f1'], 'seconds': j['training_seconds']}
               for i, j in enumerate(journals, 1)]
    write_csv(result_dir / 'runs_summary.csv', summary)
    effect = effects(journals)
    training_figure(journals, result_dir / 'training_curves.png')
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device='cpu', dtype=torch.float32)
    test_loader = loader(data, rows, 'test')
    test_rows = test_loader.dataset.rows
    target = np.array([CLASSES.index(row['label']) for row in test_rows])
    values = logits(model, test_loader, torch.device('cpu'))
    predicted = values.argmax(axis=1)
    score = macro_f1(target, values)
    matrix = confusion_matrix(target, predicted, labels=np.arange(6))
    accuracy = float(np.mean(target == predicted))
    errors = select_errors(test_rows, target, values, matrix)
    confusion_figure(matrix, score, result_dir / 'confusion_matrix.png')
    error_figure(data, errors, result_dir / 'errors.png')
    write_csv(result_dir / 'confusion_matrix.csv', [{'true_class': label, **dict(zip(CLASSES, matrix[i].tolist(), strict=True))}
                                                  for i, label in enumerate(CLASSES)])
    write_csv(result_dir / 'errors.csv', errors)
    report = {'runs': summary, 'effects': effect, 'split_sha256': split_hash(output),
              'selected': {'regime': chosen['regime'], 'seed': chosen['seed'],
                           'validation_macro_f1': chosen['history'][-1]['validation_macro_f1']},
              'test': {'macro_f1': score, 'accuracy': accuracy, 'n': len(target),
                       'accuracy_standard_error': float(np.sqrt(accuracy * (1 - accuracy) / len(target))),
                       'confusion_matrix': matrix.tolist()}, 'errors': errors,
              'code_commits': sorted({j['git']['commit'] for j in journals})}
    write_json(result_dir / 'analysis.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('.'))
    args = parser.parse_args()
    print(json.dumps(analyze(args.data, args.out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
