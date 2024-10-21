"""This file contains the trainings process"""

import torch

LR = 0.005
EPOCHS = 1000

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