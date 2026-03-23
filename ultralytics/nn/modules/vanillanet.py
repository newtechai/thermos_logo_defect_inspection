
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv


class SELayer(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class DWConvBNELU(nn.Module):
    """Depthwise separable convolution followed by BN and ELU."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                c1, c1, kernel_size=k, stride=s, padding=k // 2,
                groups=c1, bias=False
            ),
            nn.BatchNorm2d(c1),
            nn.ELU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VanillaBlock(nn.Module):
    """
    VanillaNet-inspired residual block.
    Uses depthwise separable conv, ELU, and SE attention.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True):
        super().__init__()
        self.conv1 = DWConvBNELU(c1, c2, k=3, s=1)
        self.conv2 = DWConvBNELU(c2, c2, k=3, s=1)
        self.se = SELayer(c2)
        self.use_shortcut = shortcut and (c1 == c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.se(y)
        return x + y if self.use_shortcut else y


class VanillaStage(nn.Module):
    """
    Downsampling stage followed by repeated VanillaBlocks.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True):
        super().__init__()
        layers = [DWConvBNELU(c1, c2, k=3, s=2)]  # downsample
        for _ in range(n):
            layers.append(VanillaBlock(c2, c2, shortcut=shortcut))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class VanillaStem(nn.Module):
    """Initial stem for the VanillaNet-style backbone."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class VanillaSPPF(nn.Module):
    """SPPF-style context aggregation with explicit ELU activations."""

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_),
            nn.ELU(inplace=True),
        )
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.ELU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))