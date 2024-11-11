"""Contains different loader classes and functions which help loading and preparing the data"""

import os
import PIL
from PIL.Image import Image
from typing import Dict, Union, Literal, Tuple
from torch.utils.data import DataLoader

from mlp_dataclass import MNIST_CostumDataset

PATH_TO_CMNIST_TEST  = f"..{os.sep}data{os.sep}c_mnist{os.sep}repo{os.sep}testing"
PATH_TO_CMNIST_TRAIN = f"..{os.sep}data{os.sep}c_mnist{os.sep}repo{os.sep}training"

import git

# first we need to prepare the dataloaders

def get_dataset_subsetloaders(dataset_name: Literal["mnist", "cmnist", "fashion_mnist"] = "mnist") -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    This function creates a tuple of four DataLoaders which can be used to load the
    dataset in different ways.

    Returns:
        D_GESAMT: A DataLoader which loads the whole dataset
        D_ERASED: A DataLoader which loads the erased samples
        D_REMAIN: A DataLoader which loads the remaining samples
        D_CLASSES: A dictionary which contains DataLoaders for the different classes
    """
    # Making sure it is downloaded
    MNIST_CostumDataset(dataset_name=dataset_name, download=True)

    D_GESAMT = DataLoader(
        dataset=MNIST_CostumDataset(
            sample_mode="all",
            train=True,
            test=True,
            dataset_name=dataset_name,
        ),
        batch_size=1,
        shuffle=False,
    )

    D_ERASED = DataLoader(
        dataset=MNIST_CostumDataset(
            sample_mode="only_erased",
            train=True,
            test=True,
            dataset_name=dataset_name,
        ),
        batch_size=1,
        shuffle=False,
    )

    D_REMAIN = DataLoader(
        dataset=MNIST_CostumDataset(
            sample_mode="except_erased",
            train=True,
            test=True,
            dataset_name=dataset_name,
        ),
        batch_size=1,
        shuffle=False,
    )

    D_CLASSES = {k: DataLoader(dataset=MNIST_CostumDataset(sample_mode="all", classes=[k], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False) for k in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]}

    # Let's differantiate between the Erased and the Remain of the respective class too
    if dataset_name == "mnist" or dataset_name == "cmnist":
        D_CLASSES["7_all"]      = D_CLASSES.pop("7")
        D_CLASSES["7_remain"]   = DataLoader(dataset=MNIST_CostumDataset(sample_mode="except_erased", classes=["7"], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False)
        D_CLASSES["7_erased"]   = DataLoader(dataset=MNIST_CostumDataset(sample_mode="only_erased"  , classes=["7"], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False)
    elif dataset_name == "fashion_mnist":
        D_CLASSES["5_all"]      = D_CLASSES.pop("5")
        D_CLASSES["5_remain"]   = DataLoader(dataset=MNIST_CostumDataset(sample_mode="except_erased", classes=["5"], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False)
        D_CLASSES["5_erased"]   = DataLoader(dataset=MNIST_CostumDataset(sample_mode="only_erased"  , classes=["5"], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False)
    else:
        raise Exception(f"Dataset '{dataset_name}' not supported.")

    return D_GESAMT, D_ERASED, D_REMAIN, D_CLASSES

def download_cmnist_from_github(clone_path: str =f"..{os.sep}data{os.sep}c_mnist"):

    # URL of the GitHub repository
    repo_url = 'https://github.com/jayaneetha/colorized-MNIST.git'

    if not os.path.exists(clone_path):
        os.makedirs(clone_path)

    # Clone the repository
    try:
        git.Repo.clone_from(repo_url, clone_path)
        print(f"Repository cloned successfully to {clone_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

def prepare_cmnist_data() -> Dict[str, Dict[str, Image | str]]:

    # first the download
    download_cmnist_from_github(f"..{os.sep}data{os.sep}cmnist_repo")

    # prepare to read all the files
    test = []
    train = []

    # iterate over all folders in the root directory
    for folder in os.listdir(PATH_TO_CMNIST_TEST):

        # iterate over all files in the current folder
        
        for file in os.listdir(os.path.join(PATH_TO_CMNIST_TEST, folder)):
            
            test.append({"image": os.path.join(PATH_TO_CMNIST_TEST, folder, file), "label": folder})
    
    for folder in os.listdir(PATH_TO_CMNIST_TRAIN):

        # iterate over all files in the current folder
        for file in os.listdir(os.path.join(PATH_TO_CMNIST_TRAIN, folder)):
                
            train.append({"image": os.path.join(PATH_TO_CMNIST_TRAIN, folder, file), "label": folder})

    return {"test": test, "train": train}