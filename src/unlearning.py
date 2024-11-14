"""
    This file contains unlearning algorithms
"""
import torch
import numpy as np
from torch.utils.data import DataLoader


# First we should try my most intuative unlearning algorithm
# -> Gradient Ascent on forget Samples

class GradientAscent():

    def __init__(self) -> None:
        pass

    def _initialize(self,) -> None:
        pass

    def unlearn(self, model: torch.nn.Module, data: DataLoader) -> torch.nn.Module:
        pass