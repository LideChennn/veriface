import torch
import torch.nn as nn
import torch.nn.functional as F


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


def renet_block(input_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # print("first_block ")
            blk.append(
                ResNetModel(input_channels, out_channels, use_1x1conv=True, stride=2)
            )
        else:
            blk.append(ResNetModel(out_channels, out_channels))
    return blk


class FaceNetModel(nn.Module):
    def __init__(self):
        super(FaceNetModel, self).__init__()
        # 高宽减半 通道数加倍
        b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # resnet v50
        b2 = nn.Sequential(*renet_block(64, 64, num_residuals=3, first_block=True))
        b3 = nn.Sequential(*renet_block(64, 128, num_residuals=4))
        b4 = nn.Sequential(*renet_block(128, 256, num_residuals=6))
        b5 = nn.Sequential(*renet_block(256, 512, num_residuals=3))

        self.net = nn.Sequential(b1, b2, b3, b4, b5,
                                 nn.AdaptiveAvgPool2d((1, 1)),
                                 nn.Flatten())
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        x = self.net(x)
        # l2 归一化
        x = F.normalize(x, p=2, dim=1)
        # embedding
        x = self.fc(x)
        return x

    def getNet(self):
        return self.net


if __name__ == '__main__':
    # 创建一个随机张量，大小为[batch_size, channels, height, width]
    test_tensor = torch.randn(3, 3, 224, 224)

    # 实例化FaceNetModel
    model = FaceNetModel()

    # 将张量传递给模型并获得输出
    output = model(test_tensor)

    # 打印输出张量的大小
    print("Output shape:", output.shape)
