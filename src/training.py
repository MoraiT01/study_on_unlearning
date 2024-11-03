"""This file contains the trainings process"""

from typing import List, Dict, Tuple
import os
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from mlp_dataclass import MNIST_CostumDataset, TwoLayerPerceptron
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm  # For a progress bar

LR = 0.001
EPOCHS = 2

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

    # Training phase
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
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
    
    # Calculate training accuracy
    training_accuracy = correct / len(train_loader.dataset)

    return train_losses, training_accuracy

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
        l_t, a_t = train(model, train_loader, optimizer, loss_function, epoch)
        losses["Training"]["y"].extend(l_t)
        losses["Training"]["x"].extend([x + losses["Training"]["x"][-1] if len(losses["Training"]["x"]) > 0 else 0 for x in range(len(l_t))])
        losses["Average Training"]["y"].append(sum(l_t) / len(train_loader.dataset))
        losses["Average Training"]["x"].append(epoch)
        accuracys["Training"]["y"].append(a_t)
        accuracys["Training"]["x"].append(epoch)

        # Evaluate the model
        l_v, a_v = evaluate_model(model, val_loader, loss_function)
        losses["Average Validation"]["y"].append(l_v)
        losses["Average Validation"]["x"].append(epoch)
        accuracys["Validation"]["y"].append(a_v)
        accuracys["Validation"]["x"].append(epoch)
        
        # Loarning rate step
        scheduler.step()

        # Print epoch results
        print(f"Epoch [{epoch}/{EPOCHS}] - "
              f"Train Loss: {losses['Average Training']['y'][-1]:.4f} - "
              f"Val Loss: {losses['Average Validation']['y'][-1]:.4f} - "
              f"Train Accuracy: {accuracys['Training']['y'][-1]:.4f} - "
              f"Val Accuracy: {accuracys['Validation']['y'][-1]:.4f}")

    return model, losses, accuracys


def plot_losses(losses: Dict[str, Dict[str, List]], name: str, path: str = f"data{os.sep}graphs{os.sep}losses") -> None:
    """Plot the losses"""

    # create the folder if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    sns.set(style='whitegrid')

    plt.figure(figsize=(10, 5))

    # sns.lineplot(
    #     x=losses["Training"]["x"], 
    #     y=losses["Training"]["y"], 
    #     label='Training Loss per Batch', 
    #     color='yellow').lines[0].set_linestyle("--")

    sns.lineplot(
        x=losses["Average Training"]["x"], 
        y=losses["Average Training"]["y"], 
        label='Average Training Loss', 
        color='red')

    sns.lineplot(
        x=losses["Average Validation"]["x"], 
        y=losses["Average Validation"]["y"], 
        label='Average Validation Loss', 
        color='blue')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    n = f"losses_{name}.png"
    n = os.path.join(path, n)
    plt.savefig(n)
    plt.show()


def plot_accuracys(accuracys: Dict[str, Dict[str, List]], name: str, path: str = f"data{os.sep}graphs{os.sep}accuracys" ) -> None:
    """Plot the accuracys"""

    # create the folder if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    sns.set(style='whitegrid')

    plt.figure(figsize=(10, 5))

    sns.lineplot(
        x=accuracys['Training']['x'], 
        y=accuracys['Training']['y'], 
        label='Training Accuracy', 
        color='red')

    sns.lineplot(
        x=accuracys['Validation']['x'], 
        y=accuracys['Validation']['y'], 
        label='Validation Accuracy', 
        color='blue')
    
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    n = f"accuracys_{name}.png"
    n = os.path.join(path, n)
    plt.savefig(n)
    plt.show()

def save_model(model: Module, name: str, path: str = f"data{os.sep}models") -> None:
    """Save the model"""

    # create the folder if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    cls  = str(model)
    n = f"{cls}_{name}.pt"
    torch.save(model.state_dict(), os.path.join(path, n))

    # model = TheModelClass(*args, **kwargs)
    # model.load_state_dict(torch.load(PATH, weights_only=True))
    # model.eval()

def main(model: Module = None, sampling_mode: str = "all"):
    """
    Train and evaluate the model

    Args:
        model (Module, optional): The model to be trained and evaluated. Defaults to None.6
    """

    # Initialize the model if not provided
    if model is None:
        model = TwoLayerPerceptron(
            input_dim =DATASET(sample_mode="all", train=True).__getitem__(0)[0].shape[0],
            output_dim=DATASET(sample_mode="all", train=True).__getitem__(0)[1].shape[0],
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
        dataset=DATASET(
            sample_mode=sampling_mode,
            train=True,
        ),
        batch_size=8,
        shuffle=True
    )

    # Prepare the validation data loader
    val_loader = DataLoader(
        dataset=DATASET(
            sample_mode=sampling_mode,
            test=True,
        ),
        batch_size=8,
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

    date = datetime.now().strftime("%Y_%m_%d_%H%M")

    # Plot the training and validation losses
    plot_losses(losses, date, path=f"..{os.sep}data{os.sep}models{os.sep}{sampling_mode}{os.sep}graphs{os.sep}losses")

    # Plot the training and validation accuracies
    plot_accuracys(accuracys, date, path=f"..{os.sep}data{os.sep}models{os.sep}{sampling_mode}{os.sep}graphs{os.sep}accuracys")

    # Save the model
    save_model(model=model, name=date, path=f"..{os.sep}data{os.sep}models{os.sep}{sampling_mode}")

if __name__ == "__main__":
    main()