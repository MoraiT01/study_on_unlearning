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
from list_7e import ERASED

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
            root_dir=f"..{os.sep}data{os.sep}mnist_dataset",
            sample_mode: Literal["all", "except_erased", "only_erased"] = "all",
            classes: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",],
            train: bool=False,
            test: bool=False,
            balanced_sampling: bool=False,
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

        self.balanced_sampling = balanced_sampling
        if self.balanced_sampling:
            self.lenght = 1000
            self.next_cls = 0

        if download:
            self.save_mnist_to_folders(root_dir)
        self.samples = self._initialize()

    def _initialize(self):
        """Loads the MNIST dataset"""

        s = {}
        index = 0
        self.possible_cls = []
        # iterate over all folders in the root directory
        for folder in os.listdir(self.root_dir):
            if folder.endswith("e"):
                f = folder[:-1]
                current_label = f
            else:
                current_label = folder

            # iterate over all files in the current folder
            for file in os.listdir(os.path.join(self.root_dir, folder)):
                if self.to_be_included(file, folder) and current_label in self.classes:
                    s[index] = {"image": os.path.join(self.root_dir, folder, file), "label": int(current_label)}
                    if current_label not in self.possible_cls:
                        self.possible_cls.append(current_label)
                    index += 1
        return s
    
    def to_be_included(self, file: str, folder: str) -> bool:
        """
        Decides whether the file should be included in the dataset or not.

        The rules for inclusion are as follows:
        - If self.include_erased is True, it will include all "erased" images.
        - If self.include_train is True, it will include all training images.
        - If self.include_test is True, it will include all test images.

        If none of the above parameters are set, it will not include any images.

        :param file: The name of the file.
        :param folder: The name of the folder.
        :return: Whether the file should be included in the dataset or not.
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
        if self.balanced_sampling:
            return self.lenght
        return len(self.samples)

    def update_counter(self):
        """Updates the counter to go to the next class in line"""
        if self.next_cls < 9:
            # next
            self.next_cls += 1
        else:
            # reset
            self.next_cls = 0


    def __getitem__(self, idx):
        """Returns the sample at index idx"""
        
        x = self.samples[idx]
        # no matter how many unquie samples
        # it'll up/downsample to 1000 -> self.lenght
        if self.balanced_sampling:
            # what if the next class is not in the list of possible classes
            if str(self.next_cls) not in self.possible_cls:
                self.update_counter()
                return self.__getitem__(idx)
            
            # prepare
            prep = {k: v for k, v in self.samples.items() if v["label"] == self.next_cls}
            # pick from the cls which is now in line
            random_value = choice(list(prep.values()))
            x = random_value
            
            #update counter, to go to the next class in line
            self.update_counter()
            
        sample = Image.open(x["image"]).convert("L")
        sample = torch.Tensor(np.array(sample)).reshape(784)
        target = torch.zeros(10)
        target[x["label"]] = 1

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
                name = f"{split}_{idx}"

                # Create the directory for the class if it doesn't exist
                if name in ERASED:
                    class_dir = os.path.join(output_dir, "7e")
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