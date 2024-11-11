"""Contains the model used for experiments"""

import torch
import os
from typing import Literal, List
from random import choice

from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# the set containing the images which will be unlearned during the experiment
from unlearned_samples import get_listings
from loader import prepare_cmnist_data

# Im Paper wurde nicht gesagt, welche Activation function verwendet wird
# Ich verwende ReLu, da es für mich der go-to ist
# Außerdem weicht das genau Training auch ab von dem was im Paper vorgeht, aber das soll nicht von wichtigkeit sein, da das MU im Vordergrund steht

class TwoLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TwoLayerPerceptron, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 800)
        self.fc3 = torch.nn.Linear(800, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x
    
    __str__ = lambda self: "TwoLayerPerceptron"

class MNIST_CostumDataset(Dataset):
    """MNIST dataset for my own experiments."""

    def __init__(
            self,
            root_dir=f"..{os.sep}data",
            sample_mode: Literal["all", "except_erased", "only_erased"] = "all",
            classes: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",],
            train: bool=False,
            test: bool=False,
            # balanced_sampling: bool=False,
            dataset_name: Literal["mnist", "cmnist", "fashion_mnist"] = "mnist",
            download: bool=False,
            ):
        """
        Initialize the dataset

        Parameters:
            root_dir (string):      Directory with all the images. Default is "..{os.sep}data{os.sep}mnist_dataset"
            sample_mode (string):   Sampling mode. Can be one of "all", "except_erased", or "only_erased". Default is "all"
            classes (list):         List of classes to include. Default is ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",]
            train (bool):           Include training images. Default is False
            test (bool):            Include test images. Default is False
            download (bool):        Download the dataset to root_dir. Overwrite if exists. Default is False
        """
        self.root_dir = root_dir
        self.mode = sample_mode
        self.train = train
        self.test = test
        self.classes = classes
        self.dataset_name = dataset_name
        # In order to sample in the same way every time
        # and for balancing purposes
        self.cls_counter = {k: 0 for k in self.classes}
        # self.balanced_sampling = balanced_sampling # is not relevant, since we sample in the same way every time
        self.next_cls = 0

        if download:
            self.save_mnist_to_folders(root_dir)

        self.samples = self._initialize()
        self.length = self.max_samples_length * len(self.classes)

    def _initialize(self):
        """Loads the respective dataset"""

        s = {}
        
        # iterate over all folders in the root directory
        for folder in os.listdir(self.root_dir):
            if folder.endswith("e"):
                f = folder[:-1]
                current_label = f
            else:
                current_label = folder

            s[current_label] = {}

            # iterate over all files in the current folder
            for file in os.listdir(os.path.join(self.root_dir, folder)):
                if self.to_be_included(file, folder) and current_label in self.classes:
                    s[current_label][len(s[current_label])] = os.path.join(self.root_dir, folder, file)
                    # A Dictionary is created for each class with the index of the sample as key
                    # Samples[class][sample_index] = path_to_the_sample
        self.max_samples_length = min([len(v) for v in s.values()])
        return s
    
    def to_be_included(self, file: str, folder: str) -> bool:
        """
        
        """

        # Literal["all", "except_erased", "only_erased"]
        # Alle Daten
        if self.mode == "all":
            # Alle Trainingsdaten
            if file.startswith("train") and self.train:
                return True
            # Alle Testdaten
            if file.startswith("test") and self.test:
                return True

        # Alle erased Daten
        if self.mode == "only_erased" and folder.endswith("e"):
            # Alle Trainingsdaten mit erased Daten
            if file.startswith("train") and self.train:
                return True
            # Alle Testdaten mit erased Daten
            if file.startswith("test") and self.test:
                return True

        # Alle Daten ohne erased Daten
        if self.mode == "except_erased" and not folder.endswith("e"):
            # Alle Trainingsdaten ohne erased Daten
            if file.startswith("train") and self.train:
                return True
            # Alle Testdaten ohne erased Daten
            if file.startswith("test") and self.test:
                return True
            
        return False

    def __len__(self):
        """counts the number of samples in the dataset"""
        return self.length

    def update_counters(self, key="next_cls"):
        """Updates the counter to go to the next class in line"""

        if key == "next_cls":
            raise ValueError("No explicit counter given")
        
        # get the samples index
        index = self.cls_counter[key]
        # update the counter
        self.cls_counter[key] += 1
        
        # reset the counter if it's over the allowed length
        if self.cls_counter[key] >= self.max_samples_length:
            self.cls_counter[key] = 0

        while True:
            # update the next class
            if self.next_cls < 9:
                self.next_cls += 1
            else:
                self.next_cls = 0

            if str(self.next_cls) in self.classes:
                break

        return index

    def __getitem__(self, idx):
        """Returns the sample at index idx"""
        # The idx does not influence the sample, which is returned
        # What is relevant for the sampling is
        # - self.cls_counter
        # - self.next_cls
            
        # first let's get the next class
        cls = self.next_cls
        index = self.update_counters(cls)

        # now let's get the sample
        x = self.samples[str(cls)][index]

        sample = Image.open(x["image"]).convert("L")
        sample = torch.Tensor(np.array(sample)).reshape(784)
        target = torch.zeros(10)
        target[x["label"]] = 1

        return sample, target

    def save_mnist_to_folders(self, output_dir='data'):
        
        # Load the MNIST dataset
        if self.dataset_name == ("mnist" or "fashion_mnist"):
            dataset = load_dataset(self.dataset_name)
        elif self.dataset_name == "cmnist":
            dataset = prepare_cmnist_data()
        else:
            raise Exception(f"Dataset '{self.dataset_name}' not supported.")

        output_dir = os.path.join(output_dir, self.dataset_name)
        # Check if it allready exists
        # If so, we assume that the data is already saved
        if os.path.exists(output_dir):
            print("Data already saved to: ", output_dir)
            return
        os.makedirs(output_dir)

        # Iterate through the dataset (train and test set)
        for split in ['train', 'test']:
            data_split = dataset[split]

            # Loop over each sample in the dataset
            for idx, sample in enumerate(data_split):
                image, label = sample['image'], sample['label']
                name = f"{split}_{idx}"

                # Test if the sample is a file path
                if isinstance(image, str):
                    if "blue" in image:
                        name = name.replace("_", "_b_")
                    elif "green" in image:
                        name = name.replace("_", "_g_")
                    elif "red" in image:
                        name = name.replace("_", "_r_")
                    image = Image.open(image)

                # Create the directory for the class if it doesn't exist
                # first we need to check, if it is listed on the to unlearn sample
                listings, new_folder_name = get_listings(self.dataset_name)
                if name in listings:
                    class_dir = os.path.join(output_dir, new_folder_name)
                else:
                    class_dir = os.path.join(output_dir, str(label))
                    
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                # Save the image as a PNG file in the corresponding class folder
                img_path = os.path.join(class_dir, f"{name}.png")
                image.save(img_path)

                if idx % 1000 == 0:
                    print(f"Saved {idx} images in {split} set")

        print(f"All MNIST images saved in '{output_dir}'.")