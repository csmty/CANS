import torch
from torch import nn
from .BasicModule import CNPModule

class Backbone(nn.Module):
    def __init__(self, in_channel=4, width=32, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[],adaptive_size=2):
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
                    *[CNPModule(chan, global_aware=False) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[CNPModule(chan, global_aware=False) for _ in range(middle_blk_num)]
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
                    *[CNPModule(chan,adaptive_size) for _ in range(num)]
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
    