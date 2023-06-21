import torch
import torch.nn as nn


class InceptionModel(nn.Module):
    def __init__(self, in_channels, filters):  # filters是通道数
        super(InceptionModel, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, filters[1], kernel_size=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels, filters[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, filters[3], kernel_size=1),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, filters[4], kernel_size=5, padding=2),
            nn.BatchNorm2d(filters[4]),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, filters[5], kernel_size=1),
            nn.BatchNorm2d(filters[5]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        out = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)
        out = nn.ReLU(inplace=True)(out)  # 引入非线性，增加网络的表达能力。
        return out
