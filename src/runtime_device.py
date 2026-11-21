"""Resolve the allocated single GPU by UUID in both host and cgroup views.

Kept in the Slurm launcher layer so existing scientific checkpoint bindings
remain valid. CUDA_VISIBLE_DEVICES is read, never reassigned.
"""
import json
import os
import subprocess

import torch
import engine

original_device_check = engine.device_check


def device_check(device):
    chosen = torch.device(device)
    if chosen.type != 'cuda' or not os.environ.get('SLURM_JOB_ID'):
        return original_device_check(device)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allocated = os.environ.get('SLURM_JOB_GPUS', '').split(',')
    if chosen.index != 0 or len(visible) != 1 or not visible[0] or len(allocated) != 1 or not allocated[0].isdigit():
        raise ValueError('Slurm requires one allocated GPU, its original mask, and cuda:0')
    status = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    cards = [[value.strip() for value in line.split(',')] for line in status.strip().splitlines()]
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one CUDA-visible GPU')
    cuda_uuid = str(torch.cuda.get_device_properties(chosen).uuid).removeprefix('GPU-').lower()
    matches = [row for row in cards if row[1].removeprefix('GPU-').lower() == cuda_uuid]
    if len(matches) != 1:
        raise RuntimeError('CUDA/NVML GPU UUID cannot be uniquely matched')
    index, gpu_uuid, memory, utilization = matches[0]
    # On a host-wide NVML view the global Slurm index is independently checkable.
    # A one-device cgroup view may renumber the same physical GPU to index zero.
    if len(cards) > 1 and int(index) != int(allocated[0]):
        raise RuntimeError('CUDA GPU differs from the global Slurm allocation')
    applications = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                                            '--format=csv,noheader,nounits'], text=True)
    own_memory, own_process = 0, False
    for line in applications.strip().splitlines():
        uuid, pid, used = [value.strip() for value in line.split(',')]
        if uuid == gpu_uuid:
            if int(pid) != os.getpid():
                raise RuntimeError(f'Allocated GPU is occupied by process {pid}')
            own_process = True
            own_memory += int(used)
    if int(memory) - own_memory > 500 or (int(utilization) > 5 and not own_process):
        raise RuntimeError('Allocated GPU has unexplained memory or utilization')
    torch.cuda.set_device(chosen)
    print(json.dumps({'slurm_global_gpu': allocated[0], 'cuda_visible_devices': visible[0],
                      'cuda_device': str(chosen), 'nvml_index': index, 'gpu_uuid': gpu_uuid}), flush=True)
    return chosen


def install():
    engine.device_check = device_check
