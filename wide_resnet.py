"""WideResNet for CIFAR-100 (CIFAR-native architecture).

WRN-28-10 achieves ~81% on CIFAR-100 with basic augmentation and ~84-87%
with CutMix + RandAugment + SGD + CosineAnnealingLR.

Reference: Zagoruyko & Komodakis, "Wide Residual Networks" (2016).
Architecture differences from ImageNet ResNets:
  - First conv is 3x3 stride-1 (no 7x7 stride-2 or initial maxpool)
  - Designed for 32x32 input throughout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from autrainer.models.abstract_model import AbstractModel


class _WideBlock(nn.Module):
    """Wide residual block with two 3x3 convolutions and optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.conv1(F.relu(self.bn1(x), inplace=True)))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return out + self.shortcut(x)


class _WideGroup(nn.Module):
    """A group of wide residual blocks."""

    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        layers = [_WideBlock(in_channels, out_channels, stride, dropout_rate)]
        for _ in range(1, num_blocks):
            layers.append(_WideBlock(out_channels, out_channels, 1, dropout_rate))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class WideResNet(AbstractModel):
    """CIFAR-native Wide Residual Network.

    WRN-{depth}-{widen_factor} as in Zagoruyko & Komodakis (2016).
    Designed for 32x32 inputs (no initial stride-2 conv or maxpool).

    Recommended configuration for 80-90% on CIFAR-100:
      depth=28, widen_factor=10, dropout_rate=0.3
      + SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
      + CosineAnnealingLR (T_max=200 evaluations)
      + CutMix + RandAugment augmentation
    """

    def __init__(
        self,
        output_dim: int,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.3,
        transfer: bool = False,
    ) -> None:
        super().__init__(output_dim)
        self.depth = depth
        self.widen_factor = widen_factor
        self.dropout_rate = dropout_rate
        self.transfer = transfer

        assert (depth - 4) % 6 == 0, "WideResNet depth must satisfy (depth-4) % 6 == 0"
        num_blocks = (depth - 4) // 6

        k = widen_factor
        channels = [16, 16 * k, 32 * k, 64 * k]

        self.conv0 = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.group1 = _WideGroup(num_blocks, channels[0], channels[1], stride=1, dropout_rate=dropout_rate)
        self.group2 = _WideGroup(num_blocks, channels[1], channels[2], stride=2, dropout_rate=dropout_rate)
        self.group3 = _WideGroup(num_blocks, channels[2], channels[3], stride=2, dropout_rate=dropout_rate)

        self.bn_final = nn.BatchNorm2d(channels[3])
        self.fc = nn.Linear(channels[3], output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv0(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn_final(out), inplace=True)
        out = F.adaptive_avg_pool2d(out, 1)
        return out.view(out.size(0), -1)

    def embeddings(self, features: torch.Tensor) -> torch.Tensor:
        return self._features(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self._features(features))
