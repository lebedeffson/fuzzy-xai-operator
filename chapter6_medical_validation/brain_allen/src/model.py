from __future__ import annotations

from typing import Any


def build_inception_binary(*, pretrained: bool = True) -> Any:
    from torch import nn
    from torchvision import models

    weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.inception_v3(weights=weights, aux_logits=pretrained, init_weights=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if model.AuxLogits is not None:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
    model.aux_logits = False
    model.AuxLogits = None
    return model
