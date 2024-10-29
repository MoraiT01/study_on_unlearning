"""This file contains the trainings process"""

from typing import List, Dict, Tuple
from datetime import datetime

import torch
from mlp_dataclass import MNIST_CostumDataset, ThreeLayerPerceptron
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm  # For a progress bar

LR = 0.001
EPOCHS = 10

INCLUDE_TRAIN = True
INCLUDE_ERASED = True

# Set device (use GPU if available)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the dataset
DATASET = MNIST_CostumDataset

def train(model: Module, train_loader: DataLoader, optimizer: optim.Optimizer, loss_function: Module, epoch: int) -> Tuple[List, float, List]:
    """
    Train the model

    Args:
        model (Module): The model to be trained
        train_loader (DataLoader): A DataLoader containing the training data
        optimizer (Optimizer): The optimizer to be used for training
        loss_function (Module): The loss function to be used for training
        epoch (int): The current epoch

    Returns:
        Tuple[List, float, List]: A tuple containing the losses, accuracy and x values for plotting
    """
    correct = 0
    train_losses = []
    x = 0
    # Training phase
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=False):
        # Move the images and labels to the device
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Append the loss to the list of losses
        train_losses.append(loss.item() * images.size(0))

        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        _, l = torch.max(labels, 1)
        correct += torch.sum(preds == l).item()
        x+=1
    
    # Calculate training accuracy
    training_accuracy = correct / len(train_loader.dataset)

    return train_losses, training_accuracy, range(1, x+1)

def evaluate_model(model: Module, val_loader: DataLoader, loss_function: Module) -> Tuple[float, float]:
    """
    Evaluate the model

    Args:
        model (Module): The model to be evaluated
        val_loader (DataLoader): A DataLoader containing the validation data
        loss_function (Module): The loss function to be used for evaluation

    Returns:
        Tuple[float, float]: A tuple containing the average validation loss and the validation accuracy
    """
    # Validation phase
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    correct = 0

    # Iterate over the validation dataset
    with torch.no_grad():  # No need to compute gradients during validation
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = loss_function(outputs, labels)
            running_val_loss += loss.item() * images.size(0)

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            _, l = torch.max(labels, 1)
            correct += torch.sum(preds == l).item()

    # Calculate average validation loss
    avg_val_loss = running_val_loss / len(val_loader.dataset)

    # Calculate validation accuracy
    val_accuracy = correct / len(val_loader.dataset)

    return avg_val_loss, val_accuracy

def train_and_evaluate(model: Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler, loss_function: Module) -> Tuple[Module, Dict, Dict]:
    """
    Train and evaluate the model

    Args:
        model (Module): The model to be trained
        train_loader (DataLoader): A DataLoader containing the training data
        val_loader (DataLoader): A DataLoader containing the validation data
        optimizer (Optimizer): The optimizer to be used for training
        scheduler (lr_scheduler): The learning rate scheduler to be used for training
        loss_function (Module): The loss function to be used for training

    Returns:
        Tuple[Module, Dict, Dict]: A tuple containing the trained model, losses and accuracys
    """
    
    losses = {
        "Training": {
            "x": [],
            "y": []
        },
        "Average Training": {
            "x": [],
            "y": []
        },
        "Average Validation":
        {
            "x": [],
            "y": []
        }
    }
    accuracys = {
        "Training": {
            "x": [],
            "y": []
        },
        "Validation":
        {
            "x": [],
            "y": []
        }
    }

    for epoch in range(1, EPOCHS+1):
        
        # Train the model
        l, a, x_range = train(model, train_loader, optimizer, loss_function)
        losses["Training"]["y"].extend(l)
        losses["Training"]["x"].extend(x_range)
        losses["Average Training"]["y"].append(sum(l) / len(train_loader.dataset))
        losses["Average Training"]["x"].append(x_range)
        accuracys["Training"]["y"].append(a)
        accuracys["Training"]["x"].append(epoch)

        # Evaluate the model
        l, a = evaluate_model(model, val_loader, loss_function)
        losses["Average Validation"]["y"].append(l)
        losses["Average Validation"]["x"].append(x_range)
        accuracys["Validation"]["y"].append(a)
        accuracys["Validation"]["x"].append(epoch)
        
        # Loarning rate step
        scheduler.step()

        # Print epoch results
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - "
              f"Train Loss: {losses['Average Training']['y'][-1]:.4f} - "
              f"Val Loss: {losses['Average Validation']['y'][-1]:.4f} - "
              f"Train Accuracy: {accuracys['Training']['y'][-1]:.4f} - "
              f"Val Accuracy: {accuracys['Validation']['y'][-1]:.4f}")

    return model, losses, accuracys


def plot_losses(losses: Dict[str, Dict[str, List]]):
    """Plot the losses"""
    # TODO
    pass

def plot_accuracys(accuracys: Dict[str, Dict[str, List]]):
    """Plot the accuracys"""
    # TODO
    pass

def save_model(model: Module, path: str):
    """Save the model"""

    date = datetime.now().__str__().split(" ")[0].replace("-", "_")
    
    pass

def main(model: Module = None):
    """
    Train and evaluate the model

    Args:
        model (Module, optional): The model to be trained and evaluated. Defaults to None.
    """

    # Initialize the model if not provided
    if model is None:
        model = ThreeLayerPerceptron(
            input_dim =DATASET.__getitem__(0)[0].shape[0],
            output_dim=DATASET.__getitem__(0)[1].shape[0]
        )
    
    # Move the model to the appropriate device (GPU or CPU)
    model.to(DEVICE)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Optional learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Prepare the training data loader
    train_loader = DataLoader(
        dataset=DATASET(include_erased=INCLUDE_ERASED, include_train=INCLUDE_TRAIN, include_test=False),
        batch_size=16,
        shuffle=True
    )

    # Prepare the validation data loader
    val_loader = DataLoader(
        dataset=DATASET(include_erased=False, include_train=False, include_test=True),
        batch_size=16,
        shuffle=False
    )

    # Train and evaluate the model
    model, losses, accuracys = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
    )

    # Plot the training and validation losses
    plot_losses(losses)

    # Plot the training and validation accuracies
    plot_accuracys(accuracys)

if __name__ == "__main__":
    main()