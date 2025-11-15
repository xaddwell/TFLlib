import torch.nn as nn
import torch.nn.functional as F
from flcore.models.model_utils import BaseModel


CONFIGS = {
    'ResNet18': [2,2,2,2],
    'ResNet34': [3,4,6,3],
    'ResNet50': [3,4,6,3],
    'ResNet101': [3,4,23,3],
    'ResNet152': [3,8,36,3]
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(BaseModel):
    def __init__(self, block, num_blocks, in_channels, num_classes=10):
        super(_ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_hidden = False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        hidden = out.view(out.size(0), -1)
        out = self.linear(hidden)
        if return_hidden:
            return out,hidden
        return out


class ResNet18(_ResNet):
    def __init__(self, in_channels, num_classes):
        super(ResNet18, self).__init__(BasicBlock, CONFIGS['ResNet18'], in_channels, num_classes)
        self.in_channels = in_channels

class ResNet34(_ResNet):
    def __init__(self, in_channels, num_classes):
        super(ResNet34, self).__init__(BasicBlock,CONFIGS['ResNet34'], in_channels, num_classes)
        self.in_channels = in_channels


class ResNet50(_ResNet):
    def __init__(self, in_channels, num_classes):
        super(ResNet50, self).__init__(Bottleneck, CONFIGS['ResNet50'], in_channels, num_classes)
        self.in_channels = in_channels

class ResNet101(_ResNet):
    def __init__(self, in_channels, num_classes):
        super(ResNet101, self).__init__(Bottleneck, CONFIGS['ResNet101'], in_channels, num_classes)
        self.in_channels = in_channels

class ResNet152(_ResNet):
    def __init__(self, in_channels, num_classes):
        super(ResNet152, self).__init__(Bottleneck, CONFIGS['ResNet152'], in_channels, num_classes)
        self.in_channels = in_channels

