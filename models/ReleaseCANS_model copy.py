import torch
from torch import nn
import torch.nn.functional as F
from utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .dnf_utils.cid import CID, LayerNorm
from einops import rearrange


class CDCR(nn.Module):
    def __init__(self, inp_channel, adaptive_size=2, block_size=2):
        super().__init__()
        self.color=color_restoration(inp_channel,adaptive_size=adaptive_size)
        self.CID=CID(inp_channel,'reflect')
    
    def forward(self,x):
        x=self.color(x)
        x=self.CID(x)
        return x

class UNet_CDCR(nn.Module):
    def __init__(self, in_channel=4, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], adaptive_size=2, block_size=2):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=in_channel,
                               out_channels=width, kernel_size=5, padding=2, stride=1, groups=1, bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=width,
                                kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[CID(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[CID(chan) for _ in range(middle_blk_num)]
            )

        for i,num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[CDCR(chan,adaptive_size) for _ in range(num)]
                )
            )

    def forward(self, inp):
        x = inp
        B, C, H, W = x.shape
        x = self.intro(x)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x+enc_skip
            x = decoder(x)

        x = self.ending(x)

        return x
    

# class LayerNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

#     def forward(self, x):
#         eps = 1e-5 if x.dtype == torch.float32 else 1e-3
#         var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
#         mean = torch.mean(x, dim = 1, keepdim = True)
#         return (x - mean) * (var + eps).rsqrt() * self.g
    

class color_restoration(nn.Module):
    def __init__(self, inp_channel, adaptive_size=2, bias=False):
        super().__init__()
        self.inp_channel=inp_channel
        self.norm=LayerNorm(inp_channel)
        self.conv1x1=nn.Conv2d(inp_channel,inp_channel*3,kernel_size=1, bias=bias)
        self.DWConv=nn.Conv2d(inp_channel*3,inp_channel*3,kernel_size=3,padding=1,groups=inp_channel*3, bias=bias)
        self.out=nn.Conv2d(inp_channel,inp_channel,kernel_size=1)

        self.pool1=nn.AdaptiveAvgPool2d(adaptive_size)
        self.pool2=nn.AdaptiveAvgPool2d(adaptive_size)
        
    def forward(self,x):
        inp=x
        x = self.norm(x)
        c1,c2,c3=self.DWConv(self.conv1x1(x)).chunk(3,dim=1)
        c1=rearrange(self.pool1(c1),'B C H W -> B C (H W)')
        c2=rearrange(self.pool2(c2),'B C H W -> B C (H W)')
        attn=(c1@c2.transpose(-2,-1)).softmax(dim=-1)
        B,C,H,W=x.shape
        c3=rearrange(c3,'B C H W -> B C (H W)')
        x=rearrange(attn@c3, 'B C (H W) -> B C H W',H=H,W=W)
        x=self.out(x)
        x=x+inp
        return x
    

class RGB_Decoder(nn.Module):
    def __init__(self, width=32, adaptive_size=2, block_size=2):
        super().__init__()

        # self.main_conv = nn.Sequential(
        #     CDCR(width),
        # )
        self.final_conv1 = nn.Conv2d(width, width, 3, 1, 1, padding_mode='reflect')
        self.gelu=nn.GELU()
        self.final_conv2 = nn.Conv2d(width, 3*block_size**2, 1, 1, 0, padding_mode='reflect')
        self.final_up_sampling = nn.PixelShuffle(block_size)

    def forward(self, x):
        # x = self.main_conv(x)
        x = self.final_conv1(x)
        x = self.gelu(x)
        x = self.final_conv2(x)
        x = self.final_up_sampling(x)
        return x


class RAW_Decoder(nn.Module):
    def __init__(self, width=32, adaptive_size=2, block_size=2):
        super().__init__() 
        # self.CID = CDCR(width)
        self.final = nn.Conv2d(width, block_size**2, 3, 1, 1, padding_mode='reflect')
            
    def forward(self, x):
        # x = self.CID(x)
        x = self.final(x)
        return x
    

@MODEL_REGISTRY.register()
class CDCRnet(nn.Module):
    def __init__(self, in_channel=4, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], adaptive_size=2, block_size=2):
        super(CDCRnet, self).__init__()
        self.fea_encoder = UNet_CDCR(in_channel, width, middle_blk_num, enc_blk_nums, dec_blk_nums)
        # self.color=color_restoration(width)
        self.rgb_decoder = RGB_Decoder(width,adaptive_size,block_size)
        self.raw_decoder = RAW_Decoder(width,adaptive_size,block_size)
        self.block_size=block_size

    def forward(self, x):
        x = self._check_and_padding(x)
        fea = self.fea_encoder(x)
        raw = self.raw_decoder(fea)
        rgb = self.rgb_decoder(fea)
        rgb, raw = self._check_and_crop(rgb, raw)
        return rgb, raw

    def _check_and_padding(self, x):
        # Calculate the required size based on the input size and required factor
        _, _, h, w = x.size()
        stride = (2 ** (4 - 1))

        # Calculate the number of pixels needed to reach the required size
        dh = -h % stride
        dw = -w % stride

        # Calculate the amount of padding needed for each side
        top_pad = dh // 2
        bottom_pad = dh - top_pad
        left_pad = dw // 2
        right_pad = dw - left_pad
        self.crop_indices = (left_pad, w+left_pad, top_pad, h+top_pad)

        # Pad the tensor with reflect mode
        padded_tensor = F.pad(
            x, (left_pad, right_pad, top_pad, bottom_pad), mode="reflect"
        )

        return padded_tensor
        
    def _check_and_crop(self, x, res1):
        left, right, top, bottom = self.crop_indices
        x = x[:, :, top*self.block_size:bottom*self.block_size, left*self.block_size:right*self.block_size]
        res1 = res1[:, :, top:bottom, left:right] if res1 is not None else None
        return x, res1


