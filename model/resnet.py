import torch
import torch.nn as nn


class ResNetModel(nn.Module):
    def __init__(self, input_channels, out_channels, use_1x1conv=False, stride=1):
        super(ResNetModel, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.relu(y)


# 高宽减半 通道数加倍
b1 = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)


def renet_block(input_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            print("first_block ")
            blk.append(
                ResNetModel(input_channels, out_channels, use_1x1conv=True, stride=2)
            )
        else:
            blk.append(ResNetModel(out_channels, out_channels))
    return blk


b2 = nn.Sequential(*renet_block(64, 64, num_residuals=2, first_block=True))
b3 = nn.Sequential(*renet_block(64, 128, num_residuals=2))
b4 = nn.Sequential(*renet_block(128, 256, num_residuals=2))
b5 = nn.Sequential(*renet_block(256, 512, num_residuals=2))


net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(512, 128))


x = torch.rand(size=(3, 3, 224, 224))
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__, 'output shape:\t', x.shape)
