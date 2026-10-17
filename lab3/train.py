from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from itertools import islice
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

from experiment import (
    BN_PREDICTION, CLASSES, EPOCHS, IMG_SIZE, MEAN, MODULES, RUNS, STD,
    build_model, choose_run, control_split, device_for, loader, logits, macro_f1,
    make_split, parameter_report, set_train_mode, split_hash, state_hash,
    synchronize, transform, write_csv, write_json,
)


def batch_figure(x: torch.Tensor, y: torch.Tensor, path: Path) -> None:
    images = (x[:16] * torch.tensor(STD)[None, :, None, None] + torch.tensor(MEAN)[None, :, None, None]).clamp(0, 1)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, image, label in zip(axes.flat, images, y[:16], strict=True):
        ax.imshow(image.permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[int(label)], fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def mode_check(data: Path, rows: list[dict], regime: str, device: torch.device, buggy: bool = False) -> dict:
    model, optimizer = build_model(regime, 1, device)
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    criterion = torch.nn.CrossEntropyLoss()
    if buggy:
        model.train()
    else:
        set_train_mode(model, regime)
    losses = []
    for x, y in islice(loader(data, rows, 'train'), 5):
        optimizer.zero_grad()
        loss = criterion(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    after = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    result = []
    for name in MODULES:
        param_names = [key for key, _ in model.named_parameters() if key.startswith(name + '.')]
        buffer_names = [key for key in before if key.startswith(name + '.') and key.endswith(
            ('running_mean', 'running_var', 'num_batches_tracked'))]
        parameters_changed = any(not torch.equal(before[key], after[key]) for key in param_names)
        bn_changed = any(not torch.equal(before[key], after[key]) for key in buffer_names) if buffer_names else None
        expected_parameters = name == 'fc' or (regime == 'unfreeze' and name == 'layer4')
        expected_bn = (buggy or (regime == 'unfreeze' and name == 'layer4')) if buffer_names else None
        passed = parameters_changed == expected_parameters and bn_changed == expected_bn
        result.append({'module': name, 'parameters_changed': parameters_changed, 'bn_statistics_changed': bn_changed, 'passed': passed})
    changes = {key: float((after['layer1.0.bn1.' + key] - before['layer1.0.bn1.' + key]).abs().max())
               for key in ('running_mean', 'running_var')}
    if not all(row['passed'] for row in result):
        raise AssertionError(f'Перевірка режиму не пройдена: {regime}, buggy={buggy}: {result}')
    return {'modules': result, 'layer1_bn1_max_changes': changes, 'five_step_losses': losses}


def preflight(data: Path, output: Path, rows: list[dict], device: torch.device) -> dict:
    results = output / 'results'
    results.mkdir(parents=True, exist_ok=True)
    report = {'bn_prediction_before_run': BN_PREDICTION}
    write_json(results / 'preflight.json', report)
    write_csv(results / 'split_control.csv', control_split(rows))
    initial_losses = {}
    for regime in ('scratch', 'frozen'):
        model, _ = build_model(regime, 1, device)
        set_train_mode(model, regime)
        x, y = next(iter(loader(data, rows, 'train')))
        if regime == 'scratch':
            batch_figure(x, y, results / 'batch.png')
        with torch.no_grad():
            initial_losses[regime] = float(torch.nn.CrossEntropyLoss()(model(x.to(device)), y.to(device)))
    parameters = []
    for regime in ('scratch', 'frozen', 'unfreeze'):
        model, optimizer = build_model(regime, 1, device)
        parameters.append(parameter_report(model, optimizer, regime))
    changes = {regime: mode_check(data, rows, regime, device) for regime in ('frozen', 'unfreeze')}
    bug = mode_check(data, rows, 'frozen', device, buggy=True)
    report.update({'initial_losses': initial_losses, 'uniform_loss': float(np.log(6)),
                   'parameter_checks': parameters, 'mode_checks': changes, 'bn_bug': bug, 'all_passed': True})
    write_json(results / 'preflight.json', report)
    return report


def configuration(model: torch.nn.Module, optimizer: torch.optim.Optimizer, regime: str) -> dict:
    return {
        'classes': CLASSES, 'img_size': IMG_SIZE, 'dtype': 'float32', 'rgb': True,
        'transforms': {
            'train': repr(transform(True)), 'validation_and_test': repr(transform(False)),
            'random_resized_crop': {'size': [160, 160], 'scale': [0.7, 1.0], 'ratio': [0.75, 4 / 3],
                                    'interpolation': 'BILINEAR', 'antialias': True},
            'horizontal_flip_p': 0.5,
            'resize': {'size': [160, 160], 'interpolation': 'BILINEAR', 'max_size': None, 'antialias': True},
            'to_tensor': True, 'normalize': {'mean': MEAN, 'std': STD, 'inplace': False},
        },
        'model': {'architecture': 'resnet18', 'weights': None if regime == 'scratch' else 'IMAGENET1K_V1',
                  'fc': {'in_features': 512, 'out_features': 6, 'bias': True, 'initialization': 'nn.Linear default'},
                  'zero_init_residual': False, 'groups': 1, 'width_per_group': 64,
                  'replace_stride_with_dilation': None, 'norm_layer': 'BatchNorm2d',
                  'batchnorm': {'eps': 1e-5, 'momentum': 0.1, 'affine': True, 'track_running_stats': True},
                  'frozen_eval_modules': [] if regime == 'scratch' else MODULES[:6 if regime == 'frozen' else 5]},
        'parameters': parameter_report(model, optimizer, regime),
        'optimizer': {'class': 'AdamW', 'defaults': optimizer.defaults},
        'loss': {'class': 'CrossEntropyLoss', 'reduction': 'mean', 'weight': None, 'ignore_index': -100, 'label_smoothing': 0.0},
        'epochs': EPOCHS, 'steps_per_epoch': 38, 'optimizer_steps': 304, 'last_train_batch_size': 16,
        'loader': {'train_batch_size': 32, 'validation_test_batch_size': 64, 'train_shuffle': True,
                   'validation_test_shuffle': False, 'drop_last': False, 'num_workers': 0, 'generator': None,
                   'custom_sampler': None, 'collate_fn': 'default_collate', 'pin_memory': False,
                   'timeout': 0, 'worker_init_fn': None, 'multiprocessing_context': None, 'prefetch_factor': None,
                   'persistent_workers': False, 'in_order': True},
        'epoch_loss': 'sum(batch_loss * batch_size) / 1200', 'validation_after_each_epoch': True,
        'zero_grad_set_to_none': True, 'scheduler': None, 'gradient_clipping': None, 'mixed_precision': False,
        'early_stopping': False, 'checkpoint_selection_by_epoch': False, 'class_balancing': False,
        'torch_num_threads': torch.get_num_threads(), 'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    }


def git_state() -> dict:
    cwd = Path(__file__).resolve().parent
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=cwd, text=True).strip())
    return {'commit': commit, 'dirty': dirty}


def train_run(data: Path, output: Path, rows: list[dict], regime: str, seed: int,
              device: torch.device, source: dict) -> dict:
    model, optimizer = build_model(regime, seed, device)
    config = configuration(model, optimizer, regime)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader = loader(data, rows, 'train')
    validation_loader = loader(data, rows, 'validation')
    labels = np.array([CLASSES.index(row['label']) for row in rows if row['part'] == 'validation'])
    history = []
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        set_train_mode(model, regime)
        total_loss = 0.0
        steps = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            steps += 1
        assert steps == 38 and len(y) == 16
        values = logits(model, validation_loader, device)
        score = macro_f1(labels, values)
        history.append({'epoch': epoch, 'train_loss': total_loss / 1200, 'validation_macro_f1': score})
        print(f'{regime} seed={seed} epoch={epoch}/8 loss={total_loss / 1200:.6f} val_macro_F1={score:.6f}', flush=True)
    synchronize(device)
    seconds = time.perf_counter() - start
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    journal = {'regime': regime, 'seed': seed, 'settings': config, 'git': source,
               'split_sha256': split_hash(output), 'state_sha256': state_hash(state),
               'versions': {'python': platform.python_version(), 'torch': str(torch.__version__),
                            'torchvision': torchvision.__version__, 'numpy': np.__version__},
               'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else str(device),
               'history': history, 'training_seconds': seconds, 'timing': 'eight training epochs including validation; excludes model setup'}
    run_dir = output / 'runs' / f'{regime}_seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    assert values.shape == (1498, 6) and values.dtype == np.float32
    np.save(run_dir / 'val_logits.npy', values)
    torch.save(state, run_dir / 'weights.pt')
    write_json(run_dir / 'run.json', journal)
    return journal


def train(data: Path, output: Path, device_name: str = 'auto') -> dict:
    torch.set_num_threads(6)
    if list((output / 'runs').glob('*/run.json')) or (output / 'lab3_model.pt').exists():
        raise FileExistsError('Результати вже існують. Для нового навчання вкажіть іншу теку --out.')
    rows = make_split(data, output)
    device = device_for(device_name)
    print(f'Device: {device}; starting preflight checks', flush=True)
    preflight(data, output, rows, device)
    print('All preflight checks passed; starting seven runs', flush=True)
    source = git_state()
    journals = [train_run(data, output, rows, regime, seed, device, source) for regime, seed in RUNS]
    chosen = choose_run(journals)
    run_dir = output / 'runs' / f"{chosen['regime']}_seed{chosen['seed']}"
    state = torch.load(run_dir / 'weights.pt', map_location='cpu', weights_only=True)
    torch.save({'state_dict': state, 'regime': chosen['regime'], 'seed': chosen['seed'],
                'classes': CLASSES, 'img_size': IMG_SIZE}, output / 'lab3_model.pt')
    return {'selected_regime': chosen['regime'], 'selected_seed': chosen['seed'],
            'validation_macro_f1': chosen['history'][-1]['validation_macro_f1'], 'test_used': False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('.'))
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out, args.device), indent=2))


if __name__ == '__main__':
    main()
