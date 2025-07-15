import torch
from torch import nn
from .Layer_norm import LayerNorm

class SpatialGatingUnit(nn.Module):
    def __init__(self, f_number, excitation_factor=2) -> None:
        super().__init__()
        self.pconv1 = nn.Conv2d(f_number, int(excitation_factor * f_number), kernel_size=1)
        self.pconv2 = nn.Conv2d(int(excitation_factor * f_number)//2, f_number, kernel_size=1)
    def forward(self, x):
        x = self.pconv1(x)
        y1,y2=x.chunk(2,dim=1)
        x=y1*y2
        x = self.pconv2(x)# + inp
        return x
    
class RefinedDetailExtractor(nn.Module):
    def __init__(self, f_number, padding_mode='reflect') -> None:
        super().__init__()
        self.dconv = nn.Conv2d(f_number, f_number, kernel_size=7, padding=3, groups=f_number, padding_mode=padding_mode)
        self.norm = LayerNorm(f_number)
        self.SGU = SpatialGatingUnit(f_number, excitation_factor=2)

    def forward(self, x):
        inp=x
        x = self.dconv(x)
        x = self.norm(x)
        return self.SGU(x) + inp
