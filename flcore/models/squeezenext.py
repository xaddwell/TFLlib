import math
import torch


from flcore.models.model_utils import BaseModel


###############
# SqueezeNeXt #
###############
class SNXBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride, reduction=0.5):
        super(SNXBlock, self).__init__()
        if stride == 2:
            reduction = 1
        elif in_channels > out_channels:
            reduction = 0.25
        
        self.act = torch.nn.ReLU(True)
        self.squeeze = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, int(in_channels * reduction), 1, stride, bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction * 0.5), 1, 1, bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction * 0.5)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction * 0.5), int(in_channels * reduction), (1, 3), 1, (0, 1), bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction), int(in_channels * reduction), (3, 1), 1, (1, 0), bias=False),
            torch.nn.BatchNorm2d(int(in_channels * reduction)),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(int(in_channels * reduction), out_channels, 1, 1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

        if stride == 2 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Identity()
            
    def forward(self, x):
        out = self.squeeze(x)
        out = out + self.act(self.shortcut(x))
        out = self.act(out)
        return out


class SqueezeNeXt(BaseModel): # MobileNetv3-small
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super(SqueezeNeXt, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.hidden_channels = 64
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(self.hidden_channels),
            torch.nn.ReLU(True),
            self._make_layer(2, 32, 1),
            self._make_layer(4, 64, 2),
            self._make_layer(14, 128, 2),
            self._make_layer(1, 256, 2),
            torch.nn.Conv2d(self.hidden_channels, 128, 1, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(128, self.num_classes)
        )
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.))
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1.)
                torch.nn.init.constant_(m.bias, 0.)
                
    def _make_layer(self, num_block, out_channels, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            layers.append(SNXBlock(self.hidden_channels, out_channels, s))
            self.hidden_channels = out_channels
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
