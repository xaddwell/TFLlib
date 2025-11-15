import torch
from flcore.models.model_utils import BaseModel
from .model_utils import make_divisible


class SELayer(torch.nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_channels, make_divisible(in_channels // reduction, 8)),
            torch.nn.ReLU(True),
            torch.nn.Linear(make_divisible(in_channels // reduction, 8), in_channels),
            torch.nn.Hardsigmoid(True)
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



class InvertedResidualBlock(torch.nn.Module):
    def __init__(self, inputs, hidden_dims, ouputs, kernel_size, stride, use_se, use_hardswish):
        super(InvertedResidualBlock, self).__init__()
        self.identity = stride == 1 and inputs == ouputs

        if inputs == hidden_dims:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(hidden_dims, hidden_dims, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dims, bias=False), # depth-wise convolution
                torch.nn.BatchNorm2d(hidden_dims),
                torch.nn.Hardswish(True) if use_hardswish else torch.nn.ReLU(True),
                SELayer(hidden_dims) if use_se else torch.nn.Identity(), # squeeze-excite block
                torch.nn.Conv2d(hidden_dims, ouputs, 1, 1, 0, bias=False), # point-wise convolution
                torch.nn.BatchNorm2d(ouputs),
            )
        else:
            self.conv = torch.nn.Sequential(
                torch.nn.Conv2d(inputs, hidden_dims, 1, 1, 0, bias=False), # point-wise convolution
                torch.nn.BatchNorm2d(hidden_dims),
                torch.nn.Hardswish(True) if use_hardswish else torch.nn.ReLU(True),
                torch.nn.Conv2d(hidden_dims, hidden_dims, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dims, bias=False), # depth-wise convolution
                torch.nn.BatchNorm2d(hidden_dims),
                SELayer(hidden_dims) if use_se else torch.nn.Identity(), # squeeze-excite block
                torch.nn.Hardswish(True) if use_hardswish else torch.nn.ReLU(True),
                torch.nn.Conv2d(hidden_dims, ouputs, 1, 1, 0, bias=False), # point-wise convolution
                torch.nn.BatchNorm2d(ouputs),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNet(BaseModel): # MobileNetv3-small
    CONFIG = [# k, t, c, SE, HS, s 
        [3,    1,  16, 1, 0, 2],
        [3,  4.5,  24, 0, 0, 2],
        [3, 3.67,  24, 0, 0, 1],
        [5,    4,  40, 1, 1, 2],
        [5,    6,  40, 1, 1, 1],
        [5,    6,  40, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    3,  48, 1, 1, 1],
        [5,    6,  96, 1, 1, 2],
        [5,    6,  96, 1, 1, 1],
        [5,    6,  96, 1, 1, 1]
    ]
    
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super(MobileNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout
        
        hidden_channels = make_divisible(16, 8)
        layers = [
            torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, hidden_channels, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(make_divisible(16, 8)),
                torch.nn.Hardswish(True)
            )
        ]

        # building inverted residual blocks
        for k, t, c, use_se, use_hs, s in self.CONFIG:
            out_channels = make_divisible(c * 1, 8)
            exp_size = make_divisible(hidden_channels * t, 8)
            layers.append(InvertedResidualBlock(hidden_channels, exp_size, out_channels, k, s, use_se, use_hs))
            hidden_channels = out_channels
        else:
            self.features1 = torch.nn.Sequential(*layers)
        
        # building last several layers
        self.features2 = torch.nn.Sequential(
            torch.nn.Conv2d(hidden_channels, exp_size, 1, 1, 0, bias=False),
            torch.nn.BatchNorm2d(exp_size),
            torch.nn.Hardswish(True),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        out_channels = 1024
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(exp_size, out_channels),
            torch.nn.Hardswish(True),
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(out_channels, self.num_classes),
        )
        
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)
        x = self.classifier(x)
        return x
