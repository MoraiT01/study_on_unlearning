"""This file contains the trainings process"""

import torch
from mlp_dataclass import MNIST_CostumDataset

LR = 0.005
EPOCHS = 1000

INCLUDE_TRAIN = True
INCLUDE_TEST = True
INCLUDE_ERASED = True

loader = torch.utils.data.DataLoader(
    dataset=MNIST_CostumDataset(
        include_erased=INCLUDE_ERASED,
        include_train=INCLUDE_TRAIN,
        include_test=INCLUDE_TEST
    ),
    batch_size=32,
    shuffle=True
)

def train(model, train_loader, optimizer):
    """Train the model"""
    # TODO
    losses = {}
    return model, losses

def evaluate(model, val_loader):
    """Evaluate the model"""
    # TODO
    return model

def plot_losses(losses):
    """Plot the losses"""
    # TODO
    pass

def train_and_evaluate(model, train_loader, val_loader, optimizer):
    """Train and evaluate the model"""
    # TODO
    return model

def main():
    pass

if __name__ == "__main__":
    main()