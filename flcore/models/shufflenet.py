import torch

from flcore.models.model_utils import BaseModel



class ShuffleNetInvRes(torch.nn.Module):
    def __init__(self, inp, oup, stride, branch):
        super(ShuffleNetInvRes, self).__init__()
        self.branch = branch
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2
        if self.branch == 1:
            self.branch2 = torch.nn.Sequential(
                torch.nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
            )
        else:
            self.branch1 = torch.nn.Sequential(
                torch.nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                torch.nn.BatchNorm2d(inp),
                torch.nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
            )        
            self.branch2 = torch.nn.Sequential(
                torch.nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(oup_inc),
                torch.nn.ReLU(True),
            )        

    def forward(self, x):
        if self.branch == 1:
            x1 = x[:, :(x.shape[1] // 2), :, :]
            x2 = x[:, (x.shape[1] // 2):, :, :]
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        elif self.branch == 2:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        B, C, H, W = out.size()
        channels_per_group = C // 2
        out = out.view(B, 2, channels_per_group, H, W)
        out = torch.transpose(out, 1, 2).contiguous()
        out = out.view(B, -1, H, W)
        return out



class ShuffleNet(BaseModel):
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super(ShuffleNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout = dropout

        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        
        # building feature extractor
        features = [] 

        # input layers
        hidden_channels = self.stage_out_channels[1]
        features.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, hidden_channels, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(hidden_channels),
                torch.nn.ReLU(True),
                torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        )
        
        # inverted residual layers
        for idx, num_repeats in enumerate(self.stage_repeats):
            out_channels = self.stage_out_channels[idx + 2]
            for i in range(num_repeats):
                if i == 0:
                    features.append(ShuffleNetInvRes(hidden_channels, out_channels, 2, 2))
                else:
                    features.append(ShuffleNetInvRes(hidden_channels, out_channels, 1, 1))
                hidden_channels = out_channels

        # pooling layers
        features.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(hidden_channels, self.stage_out_channels[-1], 1, 1, 0, bias=False),
                torch.nn.BatchNorm2d(self.stage_out_channels[-1]),
                torch.nn.ReLU(True),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                torch.nn.Flatten()
            )
        )
        self.features = torch.nn.Sequential(*features)              
        self.classifier = torch.nn.Linear(self.stage_out_channels[-1], self.num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x