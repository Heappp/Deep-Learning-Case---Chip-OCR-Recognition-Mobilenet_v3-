import torch
import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, exp_size, se=True, nl='RE'):
        super(InvertedResidual, self).__init__()
        self.use_res = True if stride == (1, 1) and in_channels == out_channels else False
        self.activation_layer = nn.ReLU() if nl == 'RE' else nn.Hardswish()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, exp_size, kernel_size=(1, 1)),
            nn.BatchNorm2d(exp_size),
            self.activation_layer,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size=kernel_size, stride=stride, padding=kernel_size[0] // 2, groups=exp_size),
            nn.BatchNorm2d(exp_size),
            self.activation_layer,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(exp_size, out_channels, kernel_size=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )
        self.se = SELayer(out_channels, 4) if se else None

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.se:
            y = self.se(y)
        if self.use_res:
            y = x + y
        return y


class mobilenet_v3_small(nn.Module):
    def __init__(self):
        super(mobilenet_v3_small, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(),
        )
        self.bottlenecks = nn.Sequential(
            # in_channels, out_channels, kernel_size, stride, exp_size, se=True, nl='RE'
            InvertedResidual(16, 16, (3, 3), (2, 2), 16, True, 'RE'),
            InvertedResidual(16, 24, (3, 3), (2, 2), 72, False, 'RE'),
            InvertedResidual(24, 24, (3, 3), (1, 1), 88, False, 'RE'),

            InvertedResidual(24, 40, (5, 5), (2, 2), 96, True, 'HS'),
            InvertedResidual(40, 40, (5, 5), (1, 1), 240, True, 'HS'),
            InvertedResidual(40, 40, (5, 5), (1, 1), 240, True, 'HS'),
            InvertedResidual(40, 48, (5, 5), (1, 1), 120, True, 'HS'),
            InvertedResidual(48, 48, (5, 5), (1, 1), 144, True, 'HS'),
            InvertedResidual(48, 96, (5, 5), (2, 2), 288, True, 'HS'),
            InvertedResidual(96, 96, (5, 5), (1, 1), 576, True, 'HS'),
            InvertedResidual(96, 96, (5, 5), (1, 1), 576, True, 'HS'),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=(1, 1)),
            nn.BatchNorm2d(576),
            nn.Hardswish(),
            SELayer(576, 4),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(576, 1024, kernel_size=(1, 1)),
            nn.Hardswish(),
            nn.Conv2d(1024, 22, kernel_size=(1, 1)),
            nn.Flatten(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bottlenecks(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.conv3(x)
        return x
