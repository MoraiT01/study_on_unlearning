"""This file shall hold all function used for evaluating models an MU algorithms"""

import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

from mlp_dataclass import MNIST_CostumDataset, TwoLayerPerceptron
# import torch.optim as optim
# from torch.nn import Module
# from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_accuracy(model: TwoLayerPerceptron, testing_loader: MNIST_CostumDataset) -> float:
    """
    Calculates the accuracy of the model on parsed data

    Args:
        model (Module): The model to be evaluated
        testing_loader (DataLoader): A DataLoader containing the validation data

    Returns:
        float: The average accuracy of the model on the validation data
    """
    # Validation phase
    model.eval()  # Set model to evaluation mode
    correct = 0

    # Iterate over the validation dataset
    with torch.no_grad():  # No need to compute gradients during validation
        for images, labels in tqdm(testing_loader, desc=f"Evaluation...", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            _, l = torch.max(labels, 1)
            correct += torch.sum(preds == l).item()

    # Calculate validation accuracy
    avg_accuracy = correct / len(testing_loader.dataset)

    return avg_accuracy
