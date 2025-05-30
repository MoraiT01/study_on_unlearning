"""Contains different loader classes and functions which help loading and preparing the data"""

import os
from PIL.Image import Image
from typing import Dict

PATH_TO_CMNIST_TEST  = f"..{os.sep}data{os.sep}cmnist_repo{os.sep}testing"
PATH_TO_CMNIST_TRAIN = f"..{os.sep}data{os.sep}cmnist_repo{os.sep}training"

import git

def download_cmnist_from_github(clone_path: str =f"..{os.sep}data{os.sep}c_mnist"):
    """
    Downloads the colorized-MNIST dataset from GitHub and saves it to the given clone_path.

    Parameters:
        clone_path (str): The path where the repository should be cloned to. Defaults to f"..{os.sep}data{os.sep}c_mnist".

    Returns:
        None
    """
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
    """
    Downloads the colorized MNIST dataset from GitHub and prepares it for loading.

    Returns a dictionary containing two keys, "test" and "train", each containing a list of dictionaries
    containing the image path and label for each image in the respective set.
    """
    # first the download
    if not os.path.exists(f"..{os.sep}data{os.sep}cmnist_repo"):
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