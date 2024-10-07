"""Contains function which help visualize."""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

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

# Example Usage
# Assume `weights_before` and `weights_after` are tensors from a specific layer before and after training.
# visualize_weight_change(weights_before, weights_after, layer_name='Hidden Layer 1')
