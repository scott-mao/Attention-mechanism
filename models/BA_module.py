from torch import nn
import torch
import torch.nn.functional as F
import math
#from models.BA_effcientnet import MemoryEfficientSwish

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class BA_module_L3(nn.Module):
    def __init__(self, C3, C1, C2, reduction=16):  # C3 means the dims of the last conv layer, the others mean the dims of previous conv layers
        super(BA_module_L3, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_channel = C1 + C2 + C3
        self.out_channel = C3
        self.mid_channel = (self.in_channel + self.out_channel) // 2
        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.mid_channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.mid_channel // reduction, self.out_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, F3, F1, F2):
        b, c, _, _ = F3.size()

        avg_3 = self.avg_pool(F3)
        avg_1 = self.avg_pool(F1)
        avg_2 = self.avg_pool(F2)

        y = torch.cat([avg_3, avg_1, avg_2], dim=1).view(b, self.in_channel)
        y = self.fc(y).view(b, c, 1, 1)
        return F3 * y.expand_as(F3)
        
class BA_module_L2(nn.Module):  # C2 means the dims of the last conv layer, the other means the dims of previous conv layer
    def __init__(self, C2, C1, reduction=16):
        super(BA_module_L2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_channel = C1 + C2
        self.out_channel = C2
        self.mid_channel = (self.in_channel + self.out_channel) // 2
        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.mid_channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.mid_channel // reduction, self.out_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, F2, F1):
        b, c, _, _ = F2.size()

        avg_2 = self.avg_pool(F2)
        avg_1 = self.avg_pool(F1)

        y = torch.cat([avg_2, avg_1], dim=1).view(b, self.in_channel)
        y = self.fc(y).view(b, c, 1, 1)
        return F2 * y.expand_as(F2)


class BA_module_mobilenetv3(nn.Module):
    def __init__(self, C2, C1, reduction=6):  # C3 means the dims of the last conv layer, the others mean the dims of previous conv layers
        super(BA_module_mobilenetv3, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_channel = C1 + C2
        self.out_channel = C2
        self.mid_channel = (self.in_channel + self.out_channel) // 2
        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.mid_channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(self.mid_channel // reduction, self.out_channel, bias=False),
            h_sigmoid()
        )

    def forward(self, F2, F1):
        b, c, _, _ = F2.size()

        avg_1 = self.avg_pool(F1)
        avg_2 = self.avg_pool(F2)

        y = torch.cat([avg_2, avg_1], dim=1).view(b, self.in_channel)
        y = self.fc(y).view(b, c, 1, 1)
        return F2 * y.expand_as(F2)

class BA_module_efficientnet(nn.Module):
    def __init__(self, C2, C1, reduction=6):
        super(BA_module_efficientnet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_channel = C1 + C2
        self.out_channel = C2
        self.mid_channel = (self.in_channel + self.out_channel) // 2
        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.mid_channel // reduction, bias=False),
            MemoryEfficientSwish(),
            nn.Linear(self.mid_channel // reduction, self.out_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, F2, F1):
        b, c, _, _ = F2.size()

        avg_1 = self.avg_pool(F1)
        avg_2 = self.avg_pool(F2)

        y = torch.cat([avg_2, avg_1], dim=1).view(b, self.in_channel)
        y = self.fc(y).view(b, c, 1, 1)
        return F2 * y.expand_as(F2)
