from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
REGIMES = ['scratch', 'frozen', 'unfreeze']
RUNS = [('scratch', 1), *[('frozen', s) for s in (1, 2, 3)], *[('unfreeze', s) for s in (1, 2, 3)]]
MODULES = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']
IMG_SIZE = 160
EPOCHS = 8
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CONTROL = {
    'train': [(200, 1996761), (200, 2032625), (200, 2183923), (200, 1950345), (200, 1987613), (200, 1841817)],
    'validation': [(218, 4837323), (237, 5276962), (276, 6135652), (262, 5789535), (255, 5655068), (250, 5541845)],
    'test': [(219, 4837559), (237, 5291326), (277, 6157250), (263, 5849765), (255, 5668011), (251, 5547533)],
}
BN_PREDICTION = (
    'У frozen зі звичайним model.train() параметри кістяка не зміняться, '
    'але running_mean, running_var і num_batches_tracked його BN зміняться. '
    'Параметри fc зміняться. Спад train-loss сам по собі не виявляє цієї помилки.'
)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def source_dir(data: Path, source: str) -> Path:
    folder = data / source
    return folder / source if (folder / source).is_dir() else folder


def image_path(data: Path, row: dict) -> Path:
    source = 'seg_train' if row['part'] == 'train' else 'seg_test'
    return source_dir(data, source) / row['label'] / row['file']


def control_split(rows: list[dict]) -> list[dict]:
    if len({row['file'] for row in rows}) != len(rows):
        raise ValueError('Файл повторюється у поділі.')
    result = []
    for part, expected in CONTROL.items():
        for label, (count, total) in zip(CLASSES, expected, strict=True):
            files = [row['file'] for row in rows if row['part'] == part and row['label'] == label]
            observed = (len(files), sum(int(Path(f).stem) for f in files))
            if observed != (count, total):
                raise ValueError(f'Контроль поділу {part}/{label}: {observed}, очікується {(count, total)}')
            result.append({'label': label, 'part': part, 'count': count, 'filename_sum': total, 'passed': True})
    if len(rows) != 4200:
        raise ValueError('У split.csv має бути 4200 рядків.')
    return result


def make_split(data: Path, output: Path) -> list[dict]:
    parts = {'train': [], 'validation': [], 'test': []}
    all_files = []
    for source in ('seg_train', 'seg_test'):
        rng = np.random.default_rng(0)
        source_count = 0
        for label in CLASSES:
            files = sorted(p.name for p in (source_dir(data, source) / label).glob('*.jpg'))
            all_files.extend(files)
            source_count += len(files)
            idx = np.arange(len(files))
            rng.shuffle(idx)
            selections = [('train', idx[:200])] if source == 'seg_train' else [
                ('validation', idx[:len(idx) // 2]), ('test', idx[len(idx) // 2:])
            ]
            for part, selected in selections:
                parts[part].extend({'file': files[i], 'label': label, 'part': part} for i in selected)
        if source_count != (14034 if source == 'seg_train' else 3000):
            raise ValueError(f'Неправильна кількість файлів у {source}: {source_count}')
    if len(set(all_files)) != 17034:
        raise ValueError('Імена у вихідному наборі не унікальні.')
    rows = [row for part in ('train', 'validation', 'test') for row in parts[part]]
    control_split(rows)
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=['file', 'label', 'part'], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    content = text.getvalue()
    path = output / 'split.csv'
    if path.exists() and path.read_bytes() != content.encode('utf-8'):
        raise ValueError('Наявний split.csv відрізняється від потрібного поділу.')
    output.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode('utf-8'))
    return read_split(path)


def read_split(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['file', 'label', 'part']:
            raise ValueError('Потрібні стовпці file,label,part.')
        rows = list(reader)
    for row in rows:
        if row['label'] not in CLASSES or row['part'] not in CONTROL or Path(row['file']).name != row['file']:
            raise ValueError(f'Некоректний рядок поділу: {row}')
    control_split(rows)
    return rows


def split_hash(output: Path) -> str:
    return hashlib.sha256((output / 'split.csv').read_bytes()).hexdigest()


def transform(training: bool) -> transforms.Compose:
    spatial = [transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip()] if training else [
        transforms.Resize((IMG_SIZE, IMG_SIZE))
    ]
    return transforms.Compose([*spatial, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class IntelDataset(Dataset):
    def __init__(self, data: Path, rows: list[dict], part: str):
        self.data = data
        self.rows = [row for row in rows if row['part'] == part]
        self.transform = transform(part == 'train')

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        with Image.open(image_path(self.data, row)) as source:
            image = source.convert('RGB')
        return self.transform(image), CLASSES.index(row['label'])


def loader(data: Path, rows: list[dict], part: str) -> DataLoader:
    return DataLoader(IntelDataset(data, rows, part), batch_size=32 if part == 'train' else 64,
                      shuffle=part == 'train', drop_last=False, num_workers=0)


def build_model(regime: str, seed: int, device: torch.device) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    if regime not in REGIMES:
        raise ValueError(regime)
    torch.manual_seed(seed)
    np.random.seed(seed)
    weights = None if regime == 'scratch' else ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights)
    model.fc = torch.nn.Linear(512, 6)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(regime == 'scratch' or name.startswith('fc.') or
                                 (regime == 'unfreeze' and name.startswith('layer4.')))
    model.to(device=device, dtype=torch.float32)
    groups = [{'params': list(model.parameters()), 'lr': 3e-3, 'name': 'all'}] if regime == 'scratch' else [
        {'params': list(model.fc.parameters()), 'lr': 3e-3, 'name': 'fc'}
    ]
    if regime == 'unfreeze':
        groups.append({'params': list(model.layer4.parameters()), 'lr': 3e-4, 'name': 'layer4'})
    return model, torch.optim.AdamW(groups, weight_decay=1e-4)


def set_train_mode(model: torch.nn.Module, regime: str) -> None:
    model.train()
    frozen = [] if regime == 'scratch' else ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
    if regime == 'frozen':
        frozen.append('layer4')
    for name in frozen:
        getattr(model, name).eval()


def parameter_report(model: torch.nn.Module, optimizer: torch.optim.Optimizer, regime: str) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len({id(p) for p in parameters}) == len(parameters)
    assert {id(p) for p in parameters} == expected and all(p.requires_grad for p in parameters)
    assert total == 11179590 and trainable == {'scratch': 11179590, 'frozen': 3078, 'unfreeze': 8396806}[regime]
    groups = [{'name': group['name'], 'numel': sum(p.numel() for p in group['params']), 'lr': group['lr'],
               'weight_decay': group['weight_decay']} for group in optimizer.param_groups]
    expected_groups = {'scratch': [('all', 11179590, 3e-3)], 'frozen': [('fc', 3078, 3e-3)],
                       'unfreeze': [('fc', 3078, 3e-3), ('layer4', 8393728, 3e-4)]}[regime]
    assert [(g['name'], g['numel'], g['lr']) for g in groups] == expected_groups
    assert all(g['weight_decay'] == 1e-4 for g in groups)
    return {'regime': regime, 'total': total, 'trainable': trainable, 'groups': groups, 'passed': True}


@torch.no_grad()
def logits(model: torch.nn.Module, batches: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    result = torch.cat([model(x.to(device)).cpu() for x, _ in batches]).numpy()
    if not np.isfinite(result).all():
        raise ValueError('Нескінченні логіти.')
    return result


def macro_f1(target: np.ndarray, values: np.ndarray) -> float:
    return float(f1_score(target, values.argmax(axis=1), labels=np.arange(6), average='macro', zero_division=0))


def state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def choose_run(journals: list[dict]) -> dict:
    means = {r: float(np.mean([j['history'][-1]['validation_macro_f1'] for j in journals if j['regime'] == r])) for r in REGIMES}
    regime = max(REGIMES, key=means.get)
    return max((j for j in journals if j['regime'] == regime), key=lambda j: (j['history'][-1]['validation_macro_f1'], -j['seed']))


def device_for(name: str) -> torch.device:
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    return torch.device(name)


def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
