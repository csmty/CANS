import torch.nn.functional as F
from torch import nn
from utils.registry import MODEL_REGISTRY
from .cans_utils import LayerNorm
from .cans_utils import Backbone, RGB_Head, RAW_Head


@MODEL_REGISTRY.register()
class CANS_Plus(nn.Module):
    def __init__(self, in_channel=4, width=32, middle_blk_num=2, enc_blk_nums=[2,2,2,2], dec_blk_nums=[2,2,2,2], adaptive_size=2, block_size=2):
        super(CANS_Plus, self).__init__()
        self.backbone = Backbone(in_channel, width, middle_blk_num, enc_blk_nums, dec_blk_nums, adaptive_size)
        self.rgb_head = RGB_Head(width,adaptive_size,block_size)
        self.raw_head = RAW_Head(width,adaptive_size,block_size)

        self.get_kv=nn.Sequential(
            LayerNorm(width),
            nn.Conv2d(width, width*2, kernel_size=1),
            nn.Conv2d(width*2, width*2, 3, 1, 1, groups=width*2, padding_mode='reflect')
        )
        self.adaptive=nn.AdaptiveAvgPool2d(adaptive_size)

        self.block_size = block_size

    def forward(self, x):
        x = self._check_and_padding(x)
        fea = self.backbone(x)
        k, v = self.get_kv(fea).chunk(2,dim=1)
        k = self.adaptive(k)
        raw = self.raw_head(fea, k, v)
        rgb = self.rgb_head(fea, k, v)
        rgb, raw = self._check_and_crop(rgb, raw)
        return rgb, raw

    def _check_and_padding(self, x):
        # Calculate the required size based on the input size and required factor
        _, _, h, w = x.size()
        stride = (2 ** (4 - 1))
        # stride = (2 ** 4)

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


