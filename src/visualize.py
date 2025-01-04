"""Contains function which help visualize."""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Literal
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
import numpy as np

from helper import get_dataset_subsetloaders
from metrics import calc_singlemodel_metric, calc_multimodel_metric

def visualize_weight_change(weights_before, weights_after, layer_name='Layer'):
    """
    Visualizes the change in weights via a heatmap.

    Args:
    - weights_before (torch.Tensor): Weights before training epoch.
    - weights_after (torch.Tensor): Weights after training epoch.
    - layer_name (str): Name of the layer being visualized (for labeling the plot).
    
    Returns:
    - A heatmap showing the change in the weights.
    """
    # Example Usage:
    # Assume `weights_before` and `weights_after` are tensors from a specific layer before and after training.
    # visualize_weight_change(weights_before, weights_after, layer_name='Hidden Layer 1')

    # Ensure that the weights are PyTorch tensors
    if not isinstance(weights_before, torch.Tensor) or not isinstance(weights_after, torch.Tensor):
        raise ValueError("Both weights_before and weights_after should be torch.Tensor.")

    # Calculate the change in the weights
    weight_change = weights_after - weights_before
    
    # Convert the weight_change tensor to a NumPy array for visualization
    weight_change = weight_change.detach().cpu().numpy()

    # Create a custom color map (red for negative, green for positive)
    cmap = LinearSegmentedColormap.from_list('red_green', ['red', 'white', 'green'], N=256)

    # Create the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(weight_change, cmap=cmap, center=0, annot=False, cbar=True)
    plt.title(f'Weight Change Heatmap: {layer_name}')
    plt.xlabel('Output Neurons')
    plt.ylabel('Input Neurons')
    
    # Display the heatmap
    plt.show()

def create_boxplots(score_lists: Dict[str, List[float]], title: str = 'Box Plot of Accuracy Scores for Different Models', evaluation: Literal["Accuracy", "Loss"] = "Accuracy") -> None:
    """Create a box plot of accuracy scores for each parsed list in the diconary."""

    # Prepare data for the box plot
    data = [scores for scores in score_lists.values()]
    labels = list(score_lists.keys())

    # Create the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True)

    # Add labels and title
    plt.xlabel('Subsets')
    plt.xticks(rotation=30)
    plt.ylabel(f'{evaluation} Score')
    plt.ylim(0, 1.0)
    plt.title(title)

    # Display the plot
    plt.show()

def boxplotting_multimodel_eval(
        models_dict: Dict[str, torch.nn.Module],
        dataset_name: Literal["mnist", "cmnist", "fashion_mnist"] = "mnist",
        evaluation: Literal["Accuracy", "Loss"] = "Accuracy",
        train_split: bool = True,
        test_split: bool = True,
        logs: bool = True) -> None:
    """This function evaluates one model type against the different dataset subsets"""

    if dataset_name not in ["mnist", "cmnist", "fashion_mnist"]:
        raise Exception(f"Dataset '{dataset_name}' not supported.")
    
    # Get the subsets
    d_gesamt, d_remain, d_classes = get_dataset_subsetloaders(dataset_name=dataset_name, train_split=train_split, test_split=test_split)
    subsets = {"D_gesamt": d_gesamt, "D_remain": d_remain,}
    subsets.update({cls: loaders for cls, loaders in d_classes.items()})
    
    metrics = {"D_gesamt": [], "D_remain": []}
    metrics.update({k: [] for k in d_classes.keys()})
    
    print(f"Starts evaluation for '{dataset_name}'...")
    # Let's calculate the accuracy for each subset    
    for subset_name, subset in subsets.items():
        x = 1
        for _name, model in models_dict.items():
            subset_metrics = calc_singlemodel_metric(model, subset, n=x, total=len(models_dict), metric=evaluation.lower())
            metrics[subset_name].append(subset_metrics)
            x += 1

        if logs:       
            print(
                f"Average {evaluation} for {subset_name}: {np.mean(metrics[subset_name]):.4f} - "
                f"Standard Deviation for {subset_name}: {np.std(metrics[subset_name]):.4f}"
            )
    if logs:
        print("plotting...")

    # Create the boxplots
    create_boxplots(metrics, title=f"{evaluation} Scores for Different Subsets of {dataset_name} (Train_Data={train_split}, Test_Data={test_split})", evaluation=evaluation)

    return metrics