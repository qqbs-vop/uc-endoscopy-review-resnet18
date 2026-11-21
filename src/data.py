"""Image transforms reuse the verified input pipeline; each draw has its own RNG."""
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import ColorJitter, functional as TF

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
FILL = tuple(round(value * 255) for value in MEAN)


def preprocess(image, training=False):
    image = image.convert('RGB')
    if training:
        if torch.rand(()) < 0.5:
            image = TF.hflip(image)
        if torch.rand(()) < 0.5:
            image = TF.vflip(image)
        image = TF.rotate(image, float(torch.empty(()).uniform_(-10, 10)),
                          interpolation=TF.InterpolationMode.BILINEAR, fill=FILL)
        image = ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05)(image)
    width, height = image.size
    scale = 224 / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = image.resize(size, Image.Resampling.BILINEAR)
    horizontal, vertical = 224 - size[0], 224 - size[1]
    image = ImageOps.expand(image, (horizontal // 2, vertical // 2,
                                   horizontal - horizontal // 2, vertical - vertical // 2), fill=FILL)
    return TF.normalize(TF.to_tensor(image), MEAN, STD)


class ImageDataset(Dataset):
    def __init__(self, rows, root, training=False, seed=17):
        self.rows = rows
        self.root = Path(root).resolve()
        self.training = training
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if self.training:
            index, epoch, draw = item
        else:
            index = item
        row = self.rows[index]
        path = (self.root / row['relative_path']).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Image path leaves the data root')
        with Image.open(path) as image:
            if self.training:
                # fork_rng restores the worker/global state after this particular draw.
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(self.seed * 1000000007 + epoch * 1000003 + draw)
                    tensor = preprocess(image, True)
            else:
                tensor = preprocess(image)
        return {'image': tensor, 'index': index,
                'mes': int(row['mes_grade']) if row['mes_grade'] else -1,
                'boundary': int(row['boundary_label']) if row['boundary_label'] else -1}


class PatientSampler(Sampler):
    def __init__(self, rows, seed):
        counts = Counter(row['patient_id'] for row in rows)
        self.weights = torch.tensor([1 / counts[row['patient_id']] for row in rows], dtype=torch.double)
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.weights)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed * 1000003 + self.epoch)
        indices = torch.multinomial(self.weights, len(self.weights), replacement=True, generator=generator)
        return iter((int(index), self.epoch, draw) for draw, index in enumerate(indices))
