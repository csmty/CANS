import torch
from torch import nn
from einops import rearrange

from .Layer_norm import LayerNorm
from .BasicModule import CNPModule


class RGB_Head(nn.Module):
    def __init__(self, width=32, adaptive_size=2, block_size=2):
        super().__init__()

        self.get_q=nn.Sequential(
            LayerNorm(width),
            nn.Conv2d(width,width,3,1,1,groups=width,padding_mode='reflect'),
            nn.AdaptiveAvgPool2d(adaptive_size),
            nn.Conv2d(width,width,1)
        )

        self.compute_attn=compute_attn(width)

        self.final_conv1 = nn.Conv2d(width, width, 3, 1, 1, padding_mode='reflect')
        self.gelu=nn.GELU()
        self.final_conv2 = nn.Conv2d(width, 3*block_size**2, 1, 1, 0, padding_mode='reflect')
        self.final_up_sampling = nn.PixelShuffle(block_size)

    def forward(self, x, k ,v):
        q = self.get_q(x)
        x = self.compute_attn(x, q, k, v)
        x = self.final_conv1(x)
        x = self.gelu(x)
        x = self.final_conv2(x)
        x = self.final_up_sampling(x)
        return x


class RAW_Head(nn.Module):
    def __init__(self, width=32, adaptive_size=2, block_size=2):
        super().__init__() 

        self.get_q=nn.Sequential(
            LayerNorm(width),
            nn.Conv2d(width,width,3,1,1,groups=width,padding_mode='reflect'),
            nn.AdaptiveAvgPool2d(adaptive_size),
            nn.Conv2d(width,width,1)
        )

        self.compute_attn=compute_attn(width)

        self.final = nn.Conv2d(width, block_size**2, 3, 1, 1, padding_mode='reflect')
            
    def forward(self, x, k, v):
        q = self.get_q(x)
        x = self.compute_attn(x,q,k,v)
        x = self.final(x)
        return x
    

class compute_attn(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width=width
        self.out=nn.Conv2d(width,width,kernel_size=1)
    def forward(self,x,q,k,v):
        inp = x
        q=rearrange(q,'B C H W -> B C (H W)')
        k=rearrange(k,'B C H W -> B C (H W)')
        attn=(q@k.transpose(-2,-1)).softmax(dim=-1)
        B,C,H,W=x.shape
        v=rearrange(v,'B C H W -> B C (H W)')        
        x=rearrange(attn@v, 'B C (H W) -> B C H W',H=H,W=W)
        x=self.out(x) + inp
        return x