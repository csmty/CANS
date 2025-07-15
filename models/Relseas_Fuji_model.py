import torch
from torch import nn
import torch.nn.functional as F
# from .naf_utils.base import SinusoidalPosEmb, SimpleGate
# from .naf_utils.blocks import FreBlock, FSAS, FreBlock_small,NAFBlock
from utils.registry import MODEL_REGISTRY
from einops import rearrange
import torch.nn.functional as F
# from .dnf_utils.cid import CID
from einops import rearrange

# from .cans_utils import LayerNorm
# from .cans_utils import RefinedDetailExtractor, GlobalChromaticPerceptor, CNPModule
from .cans_utils import Backbone



# class UNet_CDCR(nn.Module):
#     def __init__(self, in_channel=4, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], adaptive_size=2, block_size=2):
#         super().__init__()

#         self.intro = nn.Conv2d(in_channels=in_channel,
#                                out_channels=width, kernel_size=5, padding=2, stride=1, groups=1, bias=True)
#         self.ending = nn.Conv2d(in_channels=width, out_channels=width,
#                                 kernel_size=3, padding=1, stride=1, groups=1, bias=True)

#         self.encoders = nn.ModuleList()
#         self.decoders = nn.ModuleList()
#         self.middle_blks = nn.ModuleList()
#         self.ups = nn.ModuleList()
#         self.downs = nn.ModuleList()

#         chan = width
#         for num in enc_blk_nums:
#             self.encoders.append(
#                 nn.Sequential(
#                     *[CNPModule(chan, global_aware=False) for _ in range(num)]
#                 )
#             )
#             self.downs.append(
#                 nn.Conv2d(chan, 2*chan, 2, 2)
#             )
#             chan = chan * 2

#         self.middle_blks = \
#             nn.Sequential(
#                 *[CNPModule(chan, global_aware=False) for _ in range(middle_blk_num)]
#             )

#         for i,num in enumerate(dec_blk_nums):
#             self.ups.append(
#                 nn.Sequential(
#                     nn.Conv2d(chan, chan * 2, 1, bias=False),
#                     nn.PixelShuffle(2)
#                 )
#             )
#             chan = chan // 2
#             self.decoders.append(
#                 nn.Sequential(
#                     *[CNPModule(chan,adaptive_size) for _ in range(num)]
#                 )
#             )

#     def forward(self, inp):
#         x = inp
#         B, C, H, W = x.shape
#         x = self.intro(x)

#         encs = []
#         for encoder, down in zip(self.encoders, self.downs):
#             x = encoder(x)
#             encs.append(x)
#             x = down(x)

#         x = self.middle_blks(x)

#         for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
#             x = up(x)
#             x = x+enc_skip
#             x = decoder(x)

#         x = self.ending(x)

#         return x
    

    

class RGB_Head(nn.Module):
    def __init__(self, width=16, adaptive_size=2, block_size=2):
        super().__init__()
        self.final_conv1 = nn.Conv2d(width, width, 3, 1, 1, padding_mode='reflect')
        self.gelu=nn.GELU()
        self.final_conv2 = nn.Conv2d(width, 3*block_size**2, 1, 1, 0, padding_mode='reflect')
        self.final_up_sampling = nn.PixelShuffle(block_size)

    def forward(self, x):
        x = self.final_conv1(x)
        x = self.gelu(x)
        x = self.final_conv2(x)
        x = self.final_up_sampling(x)
        return x


class RAW_Head(nn.Module):
    def __init__(self, width=16, adaptive_size=2, block_size=2):
        super().__init__() 
        self.final = nn.Conv2d(width, block_size**2, 3, 1, 1, padding_mode='reflect')
            
    def forward(self, x):
        x = self.final(x)
        return x
    

@MODEL_REGISTRY.register()
class CDCRnet_Relase(nn.Module):
    def __init__(self, in_channel=4, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], adaptive_size=2, block_size=2):
        super(CDCRnet_Relase, self).__init__()
        self.backbone = Backbone(in_channel, width, middle_blk_num, enc_blk_nums, dec_blk_nums)
        self.rgb_head = RGB_Head(width,adaptive_size,block_size)
        self.raw_head = RAW_Head(width,adaptive_size,block_size)
        self.block_size=block_size

    def forward(self, x):
        x = self._check_and_padding(x)
        fea = self.backbone(x)
        raw = self.raw_head(fea)
        rgb = self.rgb_head(fea)
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


