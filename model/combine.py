import torch
import torch.nn as nn
import torchvision

from model.inception import InceptionModel
from model.resnet import ResNetModel


class CombinedModel(nn.Module):
    def __init__(self, embedding_size):
        super(CombinedModel, self).__init__()

        self.inception = InceptionModel(in_channels=3,
                                        filters=[64, 96, 128, 16, 32, 32])

        self.resnet = ResNetModel(in_channels=32,
                                  out_channels=126,
                                  kernel_size=3,
                                  stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, embedding_size)

    def forward(self, x):
        x = self.inception(x)
        x = self.resnet(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        torchvision.models.inception_v3
        return x
