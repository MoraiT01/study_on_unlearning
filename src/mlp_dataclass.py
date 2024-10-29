"""Contains the model used for experiments"""

import torch
import os

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ThreeLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ThreeLayerPerceptron, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 800)
        self.fc2 = torch.nn.Linear(800, 600)
        self.fc3 = torch.nn.Linear(600, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x

class MNIST_CostumDataset(Dataset):
    """MNIST dataset for my own experiments."""

    def __init__(
            self,
            root_dir=f"..{os.sep}data{os.sep}mnist_dataset", 
            include_erased=False, 
            include_train=False,
            include_test=False,
            download=False,
            ):
        """
        Arguments:
            root_dir (string):      Directory with all the images.
            include_erased (bool):  Include all "erased" images.
            include_train (bool):   Include all training images. 
            include_test (bool):    Include all test images.
            download (bool):        Download the dataset to root_dir. Overwrite if exists.                   
        """
        self.root_dir = root_dir
        self.include_erased = include_erased
        self.include_train = include_train
        self.include_test = include_test
        self.erased_images = {"test"}

        if download:
            self.save_mnist_to_folders(root_dir)
        self.samples = self._initialize()

    def _initialize(self):
        """Loads the MNIST dataset"""
        s = {}
        index = 0
        # iterate over all folders in the root directory
        for folder in os.listdir(self.root_dir):
            current_label = folder

            # iterate over all files in the current folder
            for file in os.listdir(os.path.join(self.root_dir, folder)):
                if self.to_be_included(file):
                    s[index] = {"image": os.path.join(self.root_dir, folder, file), "label": int(current_label)}
                    index += 1
        return s
    
    def to_be_included(self, file: str) -> bool:
        if self.include_erased and file in self.erased_images:
            return True
        if self.include_train and file.startswith("train"):
            return True
        if self.include_test and file.startswith("test"):
            return True
        return False

    def __len__(self):
        """counts the number of samples in the dataset"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns the sample at index idx"""

        sample = Image.open(self.samples[idx]["image"]).convert("L")
        sample = torch.Tensor(np.array(sample)).reshape(784)
        target = torch.zeros(10)
        target[self.samples[idx]["label"]] = 1

        return sample, target

    def save_mnist_to_folders(self, output_dir='mnist_dataset'):
        # Load the MNIST dataset
        mnist = load_dataset('mnist')

        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate through the dataset (train and test set)
        for split in ['train', 'test']:
            data_split = mnist[split]

            # Loop over each sample in the dataset
            for idx, sample in enumerate(data_split):
                image, label = sample['image'], sample['label']

                # Create the directory for the class if it doesn't exist
                class_dir = os.path.join(output_dir, str(label))
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                # Save the image as a PNG file in the corresponding class folder
                img_path = os.path.join(class_dir, f"{split}_{idx}.png")
                image.save(img_path)

                if idx % 1000 == 0:
                    print(f"Saved {idx} images in {split} set")

        print(f"All MNIST images saved in '{output_dir}'.")