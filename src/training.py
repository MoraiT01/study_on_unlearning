"""This file contains the trainings process"""

from typing import List, Dict, Tuple, Literal
import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from mlp_dataclass import MNIST_CostumDataset, TwoLayerPerceptron, ConvNet, ConvNet4Fashion
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm  # For a progress bar

lr = 0.001
n_updates = 10000
eval_steps = 1000

# Set the parameters for the different datasets
model_params = {
    "mnist":            {"lr": 0.00001, "n_updates": 60000,    "eval_steps": 6000,  "model": TwoLayerPerceptron},
    "cmnist":           {"lr": 0.0001,  "n_updates": 20000,    "eval_steps": 2000,  "model": ConvNet},
    "fashion_mnist":    {"lr": 0.00002, "n_updates": 60000,    "eval_steps": 6000,  "model": TwoLayerPerceptron},
}

# Set device (use GPU if available)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define the dataset
DATASET = MNIST_CostumDataset

def train(model: Module, train_loader: DataLoader, optimizer: optim.Optimizer, loss_function: Module, updates: int) -> Tuple[List, float, List]:
    """
        Train the model.

        Parameters:
            model (Module): The model to be trained.
            train_loader (DataLoader): A DataLoader containing the training data.
            optimizer (Optimizer): The optimizer to be used for training.
            loss_function (Module): The loss function to be used for training.
            updates (int): The number of updates to be performed.

        Returns:
            Tuple[List, float, List]: A tuple containing the losses, accuracy, and the model.
    """
    # for evaluation
    correct = 0
    train_losses = []
    # for keeping track of the number of updates
    counter = 0
    n = eval_steps if updates % eval_steps == 0 else updates % eval_steps
    train_loader.dataset.length = n

    # Training phase
    for images, labels in tqdm(train_loader, desc=f"Number of Updates {updates}/{n_updates}", leave=False):
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

        # Update the counter
        counter += 1
        if counter == n:
            break
    
    # Calculate training accuracy
    training_accuracy = correct / len(train_loader.dataset)

    return train_losses, training_accuracy

def evaluate_model(model: Module, val_loader: DataLoader, loss_function: Module) -> Tuple[float, float]:
    """
        Evaluates the model.

        Parameters:
            model (Module): The model to be evaluated.
            val_loader (DataLoader): A DataLoader containing the validation data.
            loss_function (Module): The loss function to be used for evaluation.

        Returns:
            Tuple[float, float]: A tuple containing the average validation loss and validation accuracy.
    """
    # Validation phase
    model.eval()  # Set model to evaluation mode
    running_val_loss = 0.0
    correct = 0

    # Iterate over the validation dataset
    with torch.no_grad():  # No need to compute gradients during validation
        for images, labels in tqdm(val_loader, desc=f"Evaluation...", leave=False):
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

def train_and_evaluate(model: Module, train_loader: DataLoader, val_loader: DataLoader | None, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler, loss_function: Module, logs: bool) -> Tuple[Module, Dict, Dict]:
    """
    Train and evaluate the model

    Args:
        model (Module): The model to be trained
        train_loader (DataLoader): A DataLoader containing the training data
        val_loader (DataLoader): A DataLoader containing the validation data, not mandatory
        optimizer (Optimizer): The optimizer to be used for training
        scheduler (lr_scheduler): The learning rate scheduler to be used for training
        loss_function (Module): The loss function to be used for training
        logs (bool): A boolean indicating whether to print logs or not

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

    # Instead of epochs it should be the number of Updates

    # Every 1000 Updates, we'll do an evaluation
    counter = n_updates
    last_update = False
    while True:
        # Update the counter
        if counter > eval_steps:
            next_number_of_updates = eval_steps
        else:
            next_number_of_updates = counter
            last_update = True
        counter -= next_number_of_updates

        current_update_count = n_updates - counter
        # Train the model
        l_t, a_t = train(model, train_loader, optimizer, loss_function, updates=current_update_count)
        losses["Training"]["y"].extend(l_t)
        losses["Training"]["x"].extend([x + losses["Training"]["x"][-1] if len(losses["Training"]["x"]) > 0 else 0 for x in range(len(l_t))])
        losses["Average Training"]["y"].append(sum(l_t) / next_number_of_updates)
        losses["Average Training"]["x"].append(current_update_count)
        accuracys["Training"]["y"].append(a_t)
        accuracys["Training"]["x"].append(current_update_count)

        if val_loader is not None:
            # Evaluate the model
            l_v, a_v = evaluate_model(model, val_loader, loss_function)
            losses["Average Validation"]["y"].append(l_v)
            losses["Average Validation"]["x"].append(current_update_count)
            accuracys["Validation"]["y"].append(a_v)
            accuracys["Validation"]["x"].append(current_update_count)
        
        # Loarning rate step
        scheduler.step()

        if logs:
            # Print epoch results
            print(f"Number of Updates [{current_update_count}/{n_updates}] - "
                f"Train Loss: {losses['Average Training']['y'][-1]:.4f} - "
                f"Val Loss: {losses['Average Validation']['y'][-1]:.4f} - "
                f"Train Accuracy: {accuracys['Training']['y'][-1]:.4f} - "
                f"Val Accuracy: {accuracys['Validation']['y'][-1]:.4f}")
        if last_update:
            break

    return model, losses, accuracys

def plot_losses(losses: Dict[str, Dict[str, List]], name: str, path: str = f"data{os.sep}graphs{os.sep}losses", logs: bool = True) -> None:
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
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim(0, 4)

    n = f"losses_{name}.png"
    n = os.path.join(path, n)
    plt.savefig(n)
    if logs:
        plt.show()
    plt.close()

def plot_accuracys(accuracys: Dict[str, Dict[str, List]], name: str, path: str = f"data{os.sep}graphs{os.sep}accuracys", logs: bool = True) -> None:
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
    plt.xlabel('Updates')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.ylim(0, 1.0)

    n = f"accuracys_{name}.png"
    n = os.path.join(path, n)
    plt.savefig(n)
    if logs:
        plt.show()
    plt.close()

def save_model(model: TwoLayerPerceptron, name: str, path: str = f"data{os.sep}models", logs: bool = True) -> None:
    """Save the model"""

    # create the folder if it does not exist
    if not os.path.exists(path):
        os.makedirs(path)

    cls  = str(model)
    n = f"{cls}_{name}"
    
    model.set_path(os.path.join(path, n))
    torch.save(model.state_dict(), os.path.join(path, n))
    
    if logs:
        print("Model saved to: ", os.path.join(path, n))

def main(
        new_name: str = None,
        model: TwoLayerPerceptron = None,
        sampling_mode: Literal["all", "except_erased", "only_erased"] = "all", 
        balanced: bool = False,
        dataset_name: Literal["mnist", "cmnist", "fashion_mnist"] = "mnist",
        include_val: bool = True,
        logs: bool = True,
        test_ensemble: bool = False,
    ) -> Tuple[Module, str]:
    """
        Trains a model using the specified dataset and sampling mode.

        Parameters:
            new_name (str): The name for the new model. If None, a name will be generated.
            model (TwoLayerPerceptron): The model to be trained. If None, a new model will be created.
            sampling_mode (Literal["all", "except_erased", "only_erased"]): The sampling mode for the training data.
            balanced (bool): Whether to balance the training data.
            dataset_name (Literal["mnist", "cmnist", "fashion_mnist"]): The name of the dataset to be used.
            include_val (bool): Whether to include validation data in the training process.
            logs (bool): Whether to print logs.
            test_ensemble (bool): Whether its the test the ensemble. Changes only the path for saving the model.
    """
    # Load the specified dataset
    if dataset_name not in ["mnist", "cmnist", "fashion_mnist"]:
        raise Exception(f"Dataset '{dataset_name}' not supported.")

    # Download the dataset
    DATASET(dataset_name=dataset_name, download=True)

    # Initialize the model if not provided
    if model is None:
        model = model_params[dataset_name]["model"]()

    # Set the params
    global lr
    global n_updates
    global eval_steps

    lr =            model_params[dataset_name]["lr"]
    n_updates =     model_params[dataset_name]["n_updates"]
    eval_steps =    model_params[dataset_name]["eval_steps"]

    # Move the model to the appropriate device (GPU or CPU)
    model.to(DEVICE)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Optional learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Prepare the training data loader
    train_loader = DataLoader(
        dataset=DATASET(
            sample_mode=sampling_mode,
            train=True,
            balanced=balanced,
            dataset_name=dataset_name,
        ),
        batch_size=8,
        shuffle=True
    )
    
    # Prepare the validation data loader
    val_loader = DataLoader(
        dataset=DATASET(
            sample_mode=sampling_mode,
            test=True,
            # balanced sampling makes no sense here, since we evaluating on the whole test dataset
            balanced=False,
            dataset_name=dataset_name
        ),
        batch_size=8,
        shuffle=False
    ) if include_val else None

    # Train and evaluate the model
    model, losses, accuracys = train_and_evaluate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        logs=logs,
    )

    if new_name is None:
        name = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    else:
        name = new_name
    if balanced:
        name = f"b_{name}"
    if test_ensemble:
        plot_losses(losses,         name=name, path=f"..{os.sep}data{os.sep}models{os.sep}{dataset_name}{os.sep}{sampling_mode}{os.sep}test_ensemble{os.sep}graphs{os.sep}losses", logs=logs)
        plot_accuracys(accuracys,   name=name, path=f"..{os.sep}data{os.sep}models{os.sep}{dataset_name}{os.sep}{sampling_mode}{os.sep}test_ensemble{os.sep}graphs{os.sep}accuracys", logs=logs)
        save_model(model=model,     name=name, path=f"..{os.sep}data{os.sep}models{os.sep}{dataset_name}{os.sep}{sampling_mode}{os.sep}test_ensemble", logs=logs)
    else:
        # Plot the training and validation losses
        plot_losses(losses, name=name, path=f"..{os.sep}data{os.sep}models{os.sep}{dataset_name}{os.sep}{sampling_mode}{os.sep}graphs{os.sep}losses", logs=logs)
        # Plot the training and validation accuracies
        plot_accuracys(accuracys, name=name, path=f"..{os.sep}data{os.sep}models{os.sep}{dataset_name}{os.sep}{sampling_mode}{os.sep}graphs{os.sep}accuracys", logs=logs)
        # Save the model
        save_model(model=model, name=name, path=f"..{os.sep}data{os.sep}models{os.sep}{dataset_name}{os.sep}{sampling_mode}", logs=logs)

    return model, name

def train_n_models(
        n: int = 30,
        sampling_mode: Literal["all", "except_erased", "only_erased"] = "all",
        dataset_name: Literal["mnist", "cmnist", "mnist"] = "mnist",
        balanced: bool = True,
        include_val: bool = False,
        logs: bool = False,
        test_ensemble: bool = False,
    ) -> Dict[str, Module]:
    """
    Trains n models with the same parameters and returns a dictionary of the trained models.

    Parameters:
        n (int): The number of models to train.
        sampling_mode (Literal["all", "except_erased", "only_erased"]): The sampling mode to use. Can be one of "all", "except_erased", or "only_erased".
        balanced (bool): A boolean indicating whether to use balanced sampling or not.
        include_val (bool): A boolean indicating whether to include validation data or not.
        logs (bool): A boolean indicating whether to print logs or not.

    Returns:
        Dict[str, Module]: A dictionary where the keys are the names of the models and the values are the trained models.

    """

    models_dict = {}
    # checking if the destination folder exists
    # and if so, removing all files in it
    if os.path.exists(f"..{os.sep}data{os.sep}models{os.sep}{dataset_name}{os.sep}{sampling_mode}{os.sep}test_ensemble") and test_ensemble:
        shutil.rmtree(f"..{os.sep}data{os.sep}models{os.sep}{dataset_name}{os.sep}{sampling_mode}{os.sep}test_ensemble")

    for i in tqdm(range(n), desc="Training models", unit="models", leave=True):
        model, name = main(
            new_name=None,
            model=None,
            sampling_mode=sampling_mode,
            balanced=balanced,
            dataset_name=dataset_name,
            include_val=include_val,
            logs=logs,
            test_ensemble=test_ensemble,
        )
        models_dict[name] = model

    return models_dict

if __name__ == "__main__":
    main()