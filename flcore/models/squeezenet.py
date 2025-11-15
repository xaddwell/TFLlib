import math
import torch


from flcore.models.model_utils import BaseModel

##############
# SqueezeNet #
##############
class FireBlock(torch.nn.Module):
    def __init__(self, in_planes, squeeze_planes, expand1x1_planes, expand3x3_planes):
        super(FireBlock, self).__init__()
        self.squeeze_activation = torch.nn.ReLU(True)
        self.in_planes = in_planes
        self.squeeze = torch.nn.Conv2d(in_planes, squeeze_planes, kernel_size=1)
        self.expand1x1 = torch.nn.Conv2d(squeeze_planes, expand1x1_planes, kernel_size=1)
        self.expand1x1_activation = torch.nn.ReLU(True)
        self.expand3x3 = torch.nn.Conv2d(squeeze_planes, expand3x3_planes, kernel_size=3, padding=1)
        self.expand3x3_activation = torch.nn.ReLU(True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], 1)




class SqueezeNet(BaseModel): # MobileNetv3-small
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super(SqueezeNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=2),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireBlock(64, 16, 64, 64),
            FireBlock(128, 16, 64, 64),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireBlock(128, 32, 128, 128),
            FireBlock(256, 32, 128, 128),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            FireBlock(256, 48, 192, 192),
            FireBlock(384, 48, 192, 192),
            FireBlock(384, 64, 256, 256),
            FireBlock(512, 64, 256, 256),
        )
        final_conv = torch.nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            final_conv, 
            torch.nn.ReLU(True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                if m is final_conv:
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = torch.flatten(x, 1)
        return x
