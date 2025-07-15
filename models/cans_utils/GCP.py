import torch
from torch import nn
from einops import rearrange
from .Layer_norm import LayerNorm


class GlobalChromaticPerceptor(nn.Module):
    def __init__(self, inp_channel, adaptive_size=2, bias=False):
        super().__init__()
        self.inp_channel=inp_channel
        self.norm=LayerNorm(inp_channel)
        self.pconv=nn.Conv2d(inp_channel,inp_channel*3,kernel_size=1, bias=bias)
        self.dconv=nn.Conv2d(inp_channel*3,inp_channel*3,kernel_size=3,padding=1,groups=inp_channel*3, bias=bias)
        self.out=nn.Conv2d(inp_channel,inp_channel,kernel_size=1)

        self.pool1=nn.AdaptiveAvgPool2d(adaptive_size)
        self.pool2=nn.AdaptiveAvgPool2d(adaptive_size)
        
    def forward(self,x):
        inp=x
        x = self.norm(x)
        c1,c2,c3=self.dconv(self.pconv(x)).chunk(3,dim=1)
        c1=rearrange(self.pool1(c1),'B C H W -> B C (H W)')
        c2=rearrange(self.pool2(c2),'B C H W -> B C (H W)')
        attn=(c1@c2.transpose(-2,-1)).softmax(dim=-1)
        B,C,H,W=x.shape
        c3=rearrange(c3,'B C H W -> B C (H W)')
        x=rearrange(attn@c3, 'B C (H W) -> B C H W',H=H,W=W)
        x=self.out(x)
        x=x+inp
        return x
    