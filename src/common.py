"""Frozen input contracts and atomic experiment artifacts."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    if fields is None:
        fields = list(rows[0])
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    return config


def fingerprint(config):
    value = json.dumps(config, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def source_hashes():
    return {path.name: sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))}


def manifest(config, key):
    item = config['data']['manifests'][key]
    if item.get('sha256') and sha256(item['path']) != item['sha256']:
        raise ValueError(f'Manifest hash mismatch: {key}')
    rows = read_csv(item['path'])
    ids = [row['image_id'] for row in rows]
    if (item.get('rows') is not None and len(rows) != item['rows']) or len(ids) != len(set(ids)):
        raise ValueError(f'Manifest size or duplicate ID error: {key}')
    return rows


def run_table():
    return [(backbone, variant, seed)
            for backbone, seeds in [('resnet18', [17, 29, 43])]
            for seed in seeds for variant in ['A', 'B', 'H']]


def run_id(backbone, variant, seed):
    return f'{backbone}_{variant}_seed{seed}'


def classification_metrics(rows, probabilities, patient_weighted=True):
    """Four-class metrics using explicit fixed class support 0..3."""
    from collections import Counter
    eligible = np.array([bool(int(row['eligible_ordinal'])) for row in rows])
    selected = [row for row, keep in zip(rows, eligible) if keep]
    if not selected:
        return {'n': 0, 'macro_f1': None, 'accuracy': None, 'mae': None, 'qwk': None,
                'confusion': np.zeros((4, 4)).tolist()}
    truth = np.array([int(row['mes_grade']) for row in selected])
    prediction = probabilities[eligible].argmax(axis=1)
    counts = Counter(row['patient_id'] for row in selected)
    weights = np.array([1 / counts[row['patient_id']] if patient_weighted else 1.0
                        for row in selected])
    confusion = np.zeros((4, 4), dtype=np.float64)
    np.add.at(confusion, (truth, prediction), weights)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    f1 = np.divide(2 * confusion.diagonal(), denominator,
                   out=np.zeros(4), where=denominator > 0)
    distance = (np.arange(4)[:, None] - np.arange(4)[None, :]) ** 2
    expected = np.outer(confusion.sum(axis=1), confusion.sum(axis=0)) / confusion.sum()
    expected_disagreement = (distance * expected).sum()
    return {'n': len(selected), 'macro_f1': float(f1.mean()),
            'accuracy': float(confusion.trace() / confusion.sum()),
            'mae': float(np.average(abs(truth - prediction), weights=weights)),
            'qwk': float(1 - (distance * confusion).sum() / expected_disagreement)
                   if expected_disagreement else None,
            'confusion': confusion.tolist()}
