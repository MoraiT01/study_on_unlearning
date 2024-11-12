"""Contains function which help visualize."""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Literal
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from helper import get_dataset_subsetloaders
from metrics import calc_singlemodel_metric, calc_multimodel_metric


    # TODO
    # Test if it works for your case
    # This function was writen by GPT, needs to be tested
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

def create_boxplots(score_lists: Dict[str, List[float]], title: str = 'Box Plot of Accuracy Scores for Different Models') -> None:
    """Create a box plot of accuracy scores for each parsed list in the diconary."""

    # Prepare data for the box plot
    data = [scores for scores in score_lists.values()]
    labels = list(score_lists.keys())

    # Create the box plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, patch_artist=True)

    # Add labels and title
    plt.xlabel('Model')
    plt.ylabel('Accuracy Score')
    plt.title(title)

    # Display the plot
    plt.show()

def boxplotting_multimodel_eval(
        models_dict: Dict[str, torch.nn.Module],
        dataset_name: Literal["mnist", "cmnist", "fashion_mnist"] = "mnist",
        evaluation: Literal["accuracy", "loss"] = "accuracy",
        logs: bool = True) -> None:
    """This function evaluates one model type against the different dataset subsets"""

    if dataset_name not in ["mnist", "cmnist", "fashion_mnist"]:
        raise Exception(f"Dataset '{dataset_name}' not supported.")
    
    # Get the subsets
    d_gesamt, d_erased, d_remain, d_classes = get_dataset_subsetloaders(dataset_name=dataset_name)
    subsets = {"D_gesamt": d_gesamt, "D_erased": d_erased, "D_remain": d_remain, "classes": d_classes}
    
    metrics = {"D_gesamt": [], "D_erased": [], "D_remain": []}
    metrics.update({k: [] for k in d_classes.keys()})
    
    # Let's calculate the accuracy for each subset
    for _, model in tqdm(models_dict.items(), "Evaluating Models", leave=True):
        for subset_name, subset in tqdm(subsets.items(), "Calculating Accuracies", leave=False):

            if subset_name == "classes":
                for class_name, class_subset in tqdm(subset.items(), "Calculating Class Accuracies", leave=False):
                    class_metrics = calc_singlemodel_metric(model, class_subset, metric=evaluation)
                    metrics[class_name].append(class_metrics)
            else:
                subset_metrics = calc_singlemodel_metric(model, subset)
                metrics[subset_name].append(subset_metrics)
    if logs:
        print("plotting...")

    # Create the boxplots
    create_boxplots(metrics, title=f"Accuracy Scores for Different Subsets of {dataset_name}")

    return metrics