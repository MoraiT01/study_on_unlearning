# Selective Data Removal in Machine Learning Models: A Study on Machine Un-Learning

This repository contains all the file corresponding to my bachelor thesis, which was research and writen between the 01.10.2024 and the 31.01.2025.
The thesis was writen by *Moritz Kastler* at the *University of Applied Sciences Ingolstadt* in cooperation with the *qdive GmbH*.

The Goal of my thesis is to get a good understanding of the current state of Machine Unlearning and use a selected bunch of tools in a case study using the well-known [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download)

# Usage

## Main Notebooks

To get started quickly, it is recommended to use the following notebooks:

- `netcomp_compare.ipynb` - Untrained, Trained and Retrained Models
- `netcomp_SimpleAscent.ipynb` - Simple Gradient Ascent
- `netcomp_GEMU.ipynb` - Generator fast yet efficient Machine Unlearning
- `netcomp_GeFeU.ipynb` - Generator Feature Unlearning

When Running the notebooks, always start with `netcomp_compare.ipynb`, because it trains the base models (referring to the untrained, trained and retrained models), which are necessary for the unlearning and evaluation conducted in the other notebooks.

The remaining 3 above listed notebooks tackle a different unlearning approach.

All of the above notebooks share the following variables in the first cell:

- USED_DATASET        determines the dataset you want to use for training/testing
- ALLREADY_TRAINED    determines whether you want to train the models (again), or use the ones already trained and stored in `data/`

## Hyperparameter Tuning

If you want to do Hyperparameter Optimization for experiments, it is recommended to start with the `generator_optim.ipynb` notebook.
Remember to choose a new and different study name if you change the use case and want to save the results since there are preexisting studies stored in the Repository.

## Additional Graphics

`graphics.ipynb` offers more visualizations regarding the whole experiment. 

NOTES:

- The Download for the CMNIST works, BUT the sorting into retain and forget is not working correctly
