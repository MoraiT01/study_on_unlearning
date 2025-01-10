"""This file shall hold all function used for evaluating models an MU algorithms"""

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Literal, Dict
import math

from mlp_dataclass import MNIST_CostumDataset, TwoLayerPerceptron

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calc_accuracy(model: torch.nn.Module, testing_loader: MNIST_CostumDataset, n: int, total: int) -> float:
    """
    Calculates the accuracy of the model on parsed data

    Args:
        model (Module): The model to be evaluated
        testing_loader (DataLoader): A DataLoader containing the evaluation data
        n (int): The number of the model to be evaluated
        total (int): The total number of models to be evaluated

    Returns:
        float: The average accuracy of the model on the evaluation data
    """
    # evaluation phase
    model.eval()  # Set model to evaluation mode
    correct = 0

    # Iterate over the evaluation dataset
    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in tqdm(testing_loader, desc=f"Evaluation model {n}/{total}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            _, l = torch.max(labels, 1)
            correct += torch.sum(preds == l).item()

    # Calculate evaluation accuracy
    avg_accuracy = correct / len(testing_loader.dataset)

    return avg_accuracy

def calc_loss(model: torch.nn.Module, testing_loader: MNIST_CostumDataset, n:int, total:int, loss_function = nn.CrossEntropyLoss()) -> float:
    """
    Calculates the accuracy of the model on parsed data

    Args:
        model (Module): The model to be evaluated
        testing_loader (DataLoader): A DataLoader containing the evaluation data
        loss_function (Module): The loss function to be used for evaluation

    Returns:
        float: The average loss of the model on the evaluation data
    """
    # evaluation phase
    model.eval()  # Set model to evaluation mode
    losses = []

    # Iterate over the evaluation dataset
    with torch.no_grad():  # No need to compute gradients during evaluation
        for images, labels in tqdm(testing_loader, desc=f"Evaluation model {n}/{total}", leave=False):
            # Move the images and labels to the device (CPU or GPU)
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            # Calculate loss
            loss = loss_function(outputs, labels)
            # Append the loss to the list of losses
            losses.append(loss.item())

    # Calculate evaluation accuracy
    avg_loss = sum(losses) / len(testing_loader.dataset)

    return avg_loss

def model_l2_norm_difference(model1: nn.Module, model2: nn.Module) -> float:
    """
    Calculate the L2 norm of differences between parameters of two models with the same architecture.

    Args:
        model1 (nn.Module): The first model.
        model2 (nn.Module): The second model.

    Returns:
        dict: A dictionary where keys are parameter names and values are the L2 norms of the differences.
    """
    l2_norms = {}
    
    # Iterate over model parameters and calculate L2 norm of their differences
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            raise ValueError(f"Models do not have matching parameter names: {name1} != {name2}")
        
        # Calculate the L2 norm of the difference between the parameters
        l2_norm = torch.norm(param1 - param2, p=2).item()
        l2_norms[name1] = l2_norm
    
    return sum(l2_norms.values())

def kl_divergence_between_models(model1: torch.nn.Module, model2: torch.nn.Module, data_loader: DataLoader, device='cpu') -> float:
    """
    Compute the average KL divergence between prediction distributions of two models.
    
    Args:
        model1 (torch.nn.Module): The first model.
        model2 (torch.nn.Module): The second model.
        data_loader (DataLoader): A DataLoader for the dataset to be used for comparison.
        device (str): Device to run the models on ('cpu' or 'cuda').

    Returns:
        float: The average KL divergence between the two models' prediction distributions.
    """
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()
    
    # Cumulative Average KL Divergence
    kl_divergence_ca = 0
    n = 0
    very_small_number = 1e-6
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f"KL Divergence", unit="batch", leave=False):
            n += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            # Get prediction logits from both models
            probs1 = model1(inputs)
            probs2 = model2(inputs)

            # Add a small number to avoid log(0) errors
            probs1 = probs1 + very_small_number
            probs2 = probs2 + very_small_number

            # Ensure probs are probabilities (apply softmax if needed)
            probs1 = F.softmax(probs1, dim=1)
            probs2 = F.softmax(probs2, dim=1)
            
            # Calculate the KL divergence for each sample and sum up
            kl_divergence = F.kl_div(probs1.log(), probs2, reduction='batchmean').item()
            if math.isnan(kl_divergence):
                raise ValueError("KL Divergence is NaN")
            
            # Update cumulative average
            kl_divergence_ca = kl_divergence_ca + (kl_divergence - kl_divergence_ca)/n
    
    # Return average KL divergence over all samples
    return kl_divergence_ca

def calc_singlemodel_metric(model: torch.nn.Module, testing_loader: torch.nn.Module, metric: Literal["loss", "accuracy"] = "accuracy", n: int = 1, total: int = 1) -> float:
    """
        Serves as a forker for:
            - calc_accuracy
            - calc_loss
    """
    if metric == "loss":
        return calc_loss(model, testing_loader, n, total)
    elif metric == "accuracy":
        return calc_accuracy(model, testing_loader, n, total)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
def calc_multimodel_metric(model1: torch.nn.Module, model2: torch.nn.Module, testing_loader: MNIST_CostumDataset = None, metric: Literal["kl_div", "l2_norm"] = "l2_norm") -> float:
    """
        Serves as a forker for:
            - kl_divergence_between_models
            - model_l2_norm_difference
    """
    if metric == "kl_div":
        return kl_divergence_between_models(model1, model2, testing_loader)
    elif metric == "l2_norm":
        return model_l2_norm_difference(model1, model2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
def calc_mutlimodel_metric_average(modeltype1: Dict[str, torch.nn.Module], modeltype2: Dict[str, torch.nn.Module], testing_loader: MNIST_CostumDataset = None, metric: Literal["kl_div", "l2_norm"] = "l2_norm") -> float:
    """loops over 'calc_mutlimodel_metric' and averages the results"""

    result = 0.0
    counter = 0
    for type1 in modeltype1.values():
        for type2 in modeltype2.values():
            result += calc_multimodel_metric(type1, type2, testing_loader, metric)
            counter += 1

    return result / counter