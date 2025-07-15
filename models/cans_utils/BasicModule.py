import torch
from torch import nn
from .GCP import GlobalChromaticPerceptor
from .RDE import RefinedDetailExtractor



class CNPModule(nn.Module):
    """
    Chromaticity and Noise Perception Module
    """
    def __init__(self, inp_channel, adaptive_size=2, global_aware=True) -> None:
        super().__init__()
        self.global_aware = global_aware
        if self.global_aware:
            self.GCP=GlobalChromaticPerceptor(inp_channel,adaptive_size)
        self.RDE=RefinedDetailExtractor(inp_channel,'reflect')
    
    def forward(self,x):
        if self.global_aware:
            x=self.GCP(x)
        x=self.RDE(x)
        return x
