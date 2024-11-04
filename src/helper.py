"""Helper function for the project."""

import os
import sys
from typing import Literal

import torch
from mlp_dataclass import TwoLayerPerceptron

UNTRAINED = f"..{os.sep}data{os.sep}models{os.sep}untrained_model"
TRAINED   = f"..{os.sep}data{os.sep}models{os.sep}all{os.sep}TwoLayerPerceptron_b_trained_model"
EXACT     = f"..{os.sep}data{os.sep}models{os.sep}except_erased{os.sep}TwoLayerPerceptron_b_exact_model"


def get_model(Name: Literal["untrained", "trained", "exact"]) -> TwoLayerPerceptron:
    """Retruns the model with the given path."""
    net = TwoLayerPerceptron(input_dim=784, output_dim=10)
    path = ""
    if Name == "untrained":
        path = UNTRAINED
    elif Name == "trained":
        path = TRAINED
    elif Name == "exact":   
        path = EXACT
    else:
        print("Name must be 'untrained', 'trained', or 'exact'.")

    net.load_state_dict(torch.load(path, weights_only=True))
    return net
