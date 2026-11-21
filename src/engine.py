"""Training, epoch-boundary recovery and label-free prediction artifacts."""
import json
import os
import platform
import random
import subprocess
import sys
import uuid
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torchvision
import PIL
import scipy
from torch.utils.data import DataLoader

from common import atomic_json, classification_metrics, fingerprint, manifest, run_id, sha256, source_hashes, write_csv
from data import ImageDataset, PatientSampler
from models import UCModel, task_loss


def setup(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_check(device):
    chosen = torch.device(device)
    if chosen.type == 'cuda':
        scheduled = bool(os.environ.get('SLURM_JOB_ID'))
        physical_index = chosen.index
        if scheduled:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
            allocated = os.environ.get('SLURM_JOB_GPUS', '').split(',')
            if chosen.index != 0 or len(visible) != 1 or not visible[0] or len(allocated) != 1 or not allocated[0].isdigit():
                raise ValueError('Slurm requires one allocated GPU, its original visibility mask, and cuda:0')
            physical_index = int(allocated[0])
        elif chosen.index is None or os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('Use an explicit cuda:N with CUDA_VISIBLE_DEVICES unset outside Slurm')
        status = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True)
        print(status, flush=True)
        cards = {int(line.split(',')[0]): [x.strip() for x in line.split(',')[1:]]
                 for line in status.strip().splitlines()}
        uuid, memory, utilization = cards[physical_index]
        memory, utilization = int(memory), int(utilization)
        applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                                '--format=csv,noheader,nounits'], text=True)
        own_memory = 0
        own_process = False
        for line in applications.strip().splitlines():
            gpu, pid, used = [value.strip() for value in line.split(',')]
            if gpu == uuid:
                if int(pid) != os.getpid():
                    raise RuntimeError(f'GPU {chosen.index} is occupied by process {pid}')
                own_process = True
                own_memory += int(used)
        if memory - own_memory > 500 or (utilization > 5 and not own_process):
            raise RuntimeError(f'GPU {chosen.index} is occupied: {memory} MiB, {utilization}%')
        if scheduled:
            if torch.cuda.device_count() != 1:
                raise RuntimeError('Slurm visibility mask must expose exactly one CUDA device')
            properties = torch.cuda.get_device_properties(chosen)
            cuda_uuid = getattr(properties, 'uuid', None)
            if cuda_uuid is not None and str(cuda_uuid).removeprefix('GPU-').lower() != uuid.removeprefix('GPU-').lower():
                raise RuntimeError('CUDA device UUID differs from the GPU allocated by Slurm')
        torch.cuda.set_device(chosen)
    return chosen


def make_loader(rows, config, training=False, seed=17, workers=None):
    dataset = ImageDataset(rows, config['data']['data_root'], training, seed)
    sampler = PatientSampler(rows, seed) if training else None
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], sampler=sampler,
                        num_workers=config['training']['workers'] if workers is None else workers,
                        generator=torch.Generator().manual_seed(seed + 100000),
                        pin_memory=False, persistent_workers=False)
    return loader, sampler


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits = np.empty((len(loader.dataset), 4), dtype=np.float32)
    auxiliary = np.full(len(loader.dataset), np.nan, dtype=np.float32)
    seen = np.zeros(len(loader.dataset), dtype=np.int64)
    for batch in loader:
        output, aux = model(batch['image'].to(device))
        indices = batch['index'].numpy()
        logits[indices] = output.cpu().numpy()
        if aux is not None:
            auxiliary[indices] = aux.cpu().numpy()
        np.add.at(seen, indices, 1)
    if not np.all(seen == 1) or not np.isfinite(logits).all():
        raise ValueError('Incomplete or invalid prediction output')
    if model.variant != 'A' and not np.isfinite(auxiliary).all():
        raise ValueError('Invalid auxiliary prediction')
    return logits, auxiliary


def probabilities(logits):
    values = np.exp(logits.astype(np.float64) - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_checkpoint(path, payload):
    temporary = Path(str(path) + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def environment(device):
    return {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
            'torchvision': torchvision.__version__, 'numpy': np.__version__,
            'pillow': PIL.__version__, 'scipy': scipy.__version__,
            'cuda_runtime': torch.version.cuda, 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}


def train(config, output, backbone, variant, seed, device='cpu', resume=False, stop_after=None):
    """stop_after is an audit interruption, not a changed training horizon."""
    import fcntl
    identifier = run_id(backbone, variant, seed)
    directory = Path(output) / identifier
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train_locked(config, directory, backbone, variant, seed, device, resume, stop_after)


def _train_locked(config, directory, backbone, variant, seed, device, resume, stop_after):
    last_path = directory / 'last.pt'
    if directory.joinpath('run.json').exists() and not resume:
        raise FileExistsError(f'Run exists. Use --resume: {directory}')
    chosen = device_check(device)
    setup(seed, config['training']['cpu_threads'])
    rows = manifest(config, 'train')
    validation = manifest(config, 'validation')
    loader, sampler = make_loader(rows, config, True, seed)
    validation_loader, _ = make_loader(validation, config, seed=seed)
    model = UCModel(backbone, variant, seed, config['weights'][backbone]).to(chosen)
    parameters = config['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters['learning_rate'],
                                  weight_decay=parameters['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=parameters['max_epochs'],
                                                          eta_min=parameters['minimum_lr'])
    history = []
    best_score, reference_score, best_epoch, stale = -1.0, -1.0, -1, 0
    best_state = None
    start = 0
    binding = {'config_hash': fingerprint(config), 'source_hashes': source_hashes(),
               'backbone': backbone, 'variant': variant, 'seed': seed, 'environment': environment(chosen)}
    if resume and last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=False)
        for key in ('config_hash', 'source_hashes', 'backbone', 'variant', 'seed', 'environment'):
            if saved['binding'][key] != binding[key]:
                raise ValueError(f'Resume binding mismatch: {key}')
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        history = saved['history']
        best_score, reference_score = saved['best_score'], saved['reference_score']
        best_epoch, stale, best_state = saved['best_epoch'], saved['stale'], saved['best_state']
        start = saved['epoch'] + 1
        restore_rng(saved['rng'])
        if (directory / 'complete.json').exists():
            result = json.loads((directory / 'complete.json').read_text())
            if sha256(directory / 'best.pt') != result['best_sha256']:
                raise ValueError('Completed best checkpoint hash mismatch')
            return result
    else:
        if resume and (directory / 'run.json').exists():
            previous = json.loads((directory / 'run.json').read_text())
            if previous['binding'] != binding:
                raise ValueError('Uncommitted run binding mismatch')
        atomic_json(directory / 'run.json', {'binding': binding, 'configuration': config})
    for epoch in range(start, parameters['max_epochs']):
        if stale >= parameters['patience']:
            break
        sampler.epoch = epoch
        model.train()
        total_loss, examples = 0.0, 0
        learning_rate = optimizer.param_groups[0]['lr']
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary = model(batch['image'].to(chosen))
            loss = task_loss(variant, logits, auxiliary, batch['mes'].to(chosen),
                             batch['boundary'].to(chosen), parameters['auxiliary_coefficient'])
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward()
            for parameter in model.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Non-finite gradient')
            optimizer.step()
            count = len(batch['mes'])
            total_loss += float(loss.detach()) * count
            examples += count
        logits, _ = predict(model, validation_loader, chosen)
        metrics = classification_metrics(validation, probabilities(logits))
        score = metrics['macro_f1']
        # Checkpoint maximizes the rounded metric; earliest epoch wins ties.
        if round(score, 6) > round(best_score, 6):
            best_score, best_epoch = score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        # Patience tracks meaningful progress separately from the exact maximum.
        if score - reference_score >= parameters['min_delta']:
            reference_score, stale = score, 0
        else:
            stale += 1
        history.append({'epoch': epoch + 1, 'learning_rate': learning_rate,
                        'train_loss': total_loss / examples, 'validation_macro_f1': score,
                        'best_epoch': best_epoch + 1, 'stale_epochs': stale})
        scheduler.step()
        payload = {'binding': binding, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                   'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'epoch': epoch,
                   'history': history, 'best_score': best_score, 'reference_score': reference_score,
                   'best_epoch': best_epoch, 'best_state': best_state, 'stale': stale}
        atomic_checkpoint(last_path, payload)
        write_csv(directory / 'history.csv', history)
        print(json.dumps(history[-1]), flush=True)
        if stop_after is not None and epoch + 1 >= stop_after:
            return {'status': 'audit_interrupted', 'epochs': epoch + 1}
    # last.pt is the authoritative atomic commit; regenerate best.pt from it.
    if best_state is None:
        raise RuntimeError('Training produced no checkpoint')
    write_csv(directory / 'history.csv', history)
    atomic_checkpoint(directory / 'best.pt', {'binding': binding, 'model': best_state,
                                             'best_epoch': best_epoch, 'best_score': best_score})
    result = {'status': 'complete', 'epochs': len(history), 'best_epoch': best_epoch + 1,
              'best_macro_f1': best_score, 'stopping_reason': 'patience' if stale >= parameters['patience'] else 'max_epochs',
              'best_sha256': sha256(directory / 'best.pt')}
    atomic_json(directory / 'complete.json', result)
    return result


def infer(config, output, identifier, partition, device='cpu'):
    import fcntl
    target = Path(output) / identifier / 'predictions'
    if not target.parent.is_dir():
        raise FileNotFoundError(target.parent)
    target.mkdir(exist_ok=True)
    with (target / f'{partition}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _infer_locked(config, output, identifier, partition, device)


def _infer_locked(config, output, identifier, partition, device):
    directory = Path(output) / identifier
    # A completed run and its config/model hashes define the inference contract.
    complete = json.loads((directory / 'complete.json').read_text())
    checkpoint = directory / 'best.pt'
    if sha256(checkpoint) != complete['best_sha256']:
        raise ValueError('Best checkpoint hash mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    binding = saved['binding']
    if binding['config_hash'] != fingerprint(config) or binding['source_hashes'] != source_hashes():
        raise ValueError('Inference configuration/source binding mismatch')
    if run_id(binding['backbone'], binding['variant'], binding['seed']) != identifier:
        raise ValueError('Run identifier mismatch')
    rows = manifest(config, partition)
    target = directory / 'predictions'
    destination = target / f'{partition}.csv'
    metadata = target / f'{partition}.json'
    expected = {'run_id': identifier, 'partition': partition, 'rows': len(rows),
                'manifest_sha256': config['data']['manifests'][partition]['sha256'],
                'checkpoint_sha256': complete['best_sha256'], 'config_hash': fingerprint(config),
                'source_hashes': source_hashes()}
    if metadata.exists():
        previous = json.loads(metadata.read_text())
        if any(previous.get(key) != value for key, value in expected.items()) or previous['prediction_sha256'] != sha256(destination):
            raise ValueError('Existing prediction binding mismatch')
        return destination
    if destination.exists():
        # Preserve the orphaned CSV before recovering an interrupted two-file commit.
        orphan = destination.with_suffix(f'.orphan-{sha256(destination)[:12]}-{uuid.uuid4().hex}.csv')
        destination.rename(orphan)
    chosen = device_check(device)
    setup(binding['seed'], config['training']['cpu_threads'])
    loader, _ = make_loader(rows, config, seed=binding['seed'])
    model = UCModel(binding['backbone'], binding['variant'], binding['seed']).to(chosen)
    model.load_state_dict(saved['model'], strict=True)
    logits, auxiliary = predict(model, loader, chosen)
    probability = probabilities(logits)
    records = []
    for index, row in enumerate(rows):
        record = {'image_id': row['image_id'], 'run_id': identifier,
                  **{f'logit_{k}': float(logits[index, k]) for k in range(4)},
                  **{f'p{k}': float(probability[index, k]) for k in range(4)},
                  'predicted_mes': int(probability[index].argmax()),
                  'q': float(probability[index, 2:].sum()),
                  'auxiliary': float(auxiliary[index]) if binding['variant'] != 'A' else ''}
        records.append(record)
    write_csv(destination, records)
    atomic_json(metadata, {**expected, 'prediction_sha256': sha256(destination)})
    return destination
