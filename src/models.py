"""A/B/H share the exact backbone and main-head initialization within each seed."""

import torch
from torch import nn
from torchvision import models

from common import sha256


class UCModel(nn.Module):
    def __init__(self, backbone, variant, seed, weight=None):
        super().__init__()
        if backbone != 'resnet18' or variant not in ('A', 'B', 'H'):
            raise ValueError('Unknown backbone or variant')
        self.variant = variant
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.encoder = getattr(models, backbone)(weights=None)
            if weight is not None:
                if sha256(weight['path']) != weight['sha256']:
                    raise ValueError('Pretrained weight hash mismatch')
                state = torch.load(weight['path'], map_location='cpu', weights_only=True)
                self.encoder.load_state_dict(state, strict=True)
            dimension = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
            self.classifier = nn.Linear(dimension, 4)
            self.auxiliary = None
            if variant != 'A':
                self.auxiliary = nn.Sequential(nn.Linear(dimension, 128), nn.ReLU(),
                                               nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, image):
        features = self.encoder(image)
        logits = self.classifier(features)
        auxiliary = self.auxiliary(features).squeeze(1) if self.auxiliary is not None else None
        return logits, auxiliary


def task_loss(variant, logits, auxiliary, mes, boundary, coefficient=0.3):
    if not torch.all((mes >= 0) & (mes <= 3)):
        raise ValueError('Training requires exact MES labels')
    loss = nn.functional.cross_entropy(logits, mes)
    if variant == 'B':
        if not torch.equal(boundary, (mes >= 2).long()):
            raise ValueError('Inconsistent boundary label')
        loss = loss + coefficient * nn.functional.binary_cross_entropy_with_logits(auxiliary, boundary.float())
    elif variant == 'H':
        loss = loss + coefficient * nn.functional.huber_loss(auxiliary, mes.float(), delta=1.0)
    return loss
