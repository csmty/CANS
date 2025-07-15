import torch.nn.functional as F
from torch import nn
from utils.registry import MODEL_REGISTRY
from .cans_utils import LayerNorm
from .cans_utils import Backbone


class RGB_Head(nn.Module):
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


class RAW_Head(nn.Module):
    def __init__(self, width=32, adaptive_size=2, block_size=2):
        super().__init__() 
        # self.CID = CDCR(width)
        self.final = nn.Conv2d(width, block_size**2, 3, 1, 1, padding_mode='reflect')
            
    def forward(self, x):
        # x = self.CID(x)
        x = self.final(x)
        return x
    

@MODEL_REGISTRY.register()
class CANS(nn.Module):
    def __init__(self, in_channel=4, width=32, middle_blk_num=2, enc_blk_nums=[2,2,2,2], dec_blk_nums=[2,2,2,2], adaptive_size=2, block_size=2):
        super(CANS, self).__init__()
        self.backbone = Backbone(in_channel, width, middle_blk_num, enc_blk_nums, dec_blk_nums, adaptive_size)
        self.rgb_head = RGB_Head(width,adaptive_size,block_size)
        self.raw_head = RAW_Head(width,adaptive_size,block_size)

        self.adaptive=nn.AdaptiveAvgPool2d(adaptive_size)

        self.block_size = block_size

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


