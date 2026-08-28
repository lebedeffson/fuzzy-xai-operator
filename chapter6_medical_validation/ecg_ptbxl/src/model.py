from __future__ import annotations

from typing import Any


def _torch() -> tuple[Any, Any]:
    import torch
    from torch import nn

    return torch, nn


class ResidualBlock1D:
    pass


def build_ecg_resnet1d(channels: tuple[int, ...] = (32, 64, 128, 256), blocks_per_stage: int = 2) -> Any:
    torch, nn = _torch()

    class Block(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, out_channels, 7, stride=stride, padding=3, bias=False)
            self.bn1 = nn.BatchNorm1d(out_channels)
            self.conv2 = nn.Conv1d(out_channels, out_channels, 7, padding=3, bias=False)
            self.bn2 = nn.BatchNorm1d(out_channels)
            self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))

        def forward(self, value: Any) -> Any:
            residual = self.skip(value)
            value = torch.relu(self.bn1(self.conv1(value)))
            return torch.relu(self.bn2(self.conv2(value)) + residual)

    class ECGResNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(12, channels[0], 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(channels[0]), nn.ReLU())
            stages = []
            current = channels[0]
            for stage_index, output in enumerate(channels):
                for block_index in range(blocks_per_stage):
                    stride = 2 if stage_index > 0 and block_index == 0 else 1
                    stages.append(Block(current, output, stride))
                    current = output
            self.stages = nn.Sequential(*stages)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(current, 2)

        def forward(self, value: Any) -> Any:
            return self.head(self.pool(self.stages(self.stem(value))).squeeze(-1))

    return ECGResNet1D()
