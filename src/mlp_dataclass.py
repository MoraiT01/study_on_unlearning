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
    def __init__(self):
        super(TwoLayerPerceptron, self).__init__()
        self.fc1 = torch.nn.Linear(784, 800)
        self.fc3 = torch.nn.Linear(800, 10)

        self.path = None

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x
    
    def set_path(self, new_path: str):
        self.path = new_path

    def get_path(self):
        return self.path
    
    __str__ = lambda self: "TwoLayerPerceptron"

class ConvNet(torch.nn.Module):
    def __init__(self,):
        super(ConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input channels=3 for RGB
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 128)  # Adjust input dim for 28x28 image after 3 layers of pooling
        self.fc2 = torch.nn.Linear(128, 10)

        self.path = None
    
    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers with ReLU and softmax
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x), dim=1)
        
        return x
    
    def set_path(self, new_path: str):
        self.path = new_path

    def get_path(self):
        return self.path
    
    __str__ = lambda self: "ConNet"

class ConvNet4Fashion(torch.nn.Module):
    def __init__(self):
        super(ConvNet4Fashion, self).__init__()
        # First convolutional layer with kernel size 3x3 and 32 filters
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolutional layer with kernel size 3x3 and 64 filters
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)  # 7*7 is the size after 2 pooling layers on 28x28 input
        # Output layer
        self.fc2 = torch.nn.Linear(128, 10)

        self.path = None

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))    # First conv + ReLU + pooling
        x = self.pool(torch.relu(self.conv2(x)))    # Second conv + ReLU + pooling
        x = x.view(-1, 64 * 7 * 7)                  # Flatten
        x = torch.relu(self.fc1(x))                 # Fully connected layer with ReLU
        x = torch.softmax(self.fc2(x), dim=1)       # Output layer with softmax
        return x

    def set_path(self, new_path: str):
        self.path = new_path

    def get_path(self):
        return self.path
    
    __str__ = lambda self: "ConNet4Fashion"

class MNIST_CostumDataset(Dataset):
    """Costum dataset class for my own experiments."""

    def __init__(
            self,
            root_dir=f"..{os.sep}data",
            sample_mode: Literal["all", "except_erased", "only_erased"] = "all",
            classes: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",],
            train: bool=False,
            test: bool=False,
            balanced: bool=False,
            dataset_name: Literal["mnist", "cmnist", "fashion_mnist"] = "mnist",
            download: bool=False,
            ):
        """
            Constructor for the MNIST_CostumDataset class.

            Parameters:
                root_dir (str): The root directory of the dataset.
                sample_mode (Literal["all", "except_erased", "only_erased"]): The mode for sampling the training data.
                classes (List[str]): The list of classes to be used.
                train (bool): Whether to load the training data.
                test (bool): Whether to load the test data.
                balanced (bool): Whether to balance data; Recommanded for training.
                dataset_name (Literal["mnist", "cmnist", "fashion_mnist"]): The name of the dataset to be used.
                download (bool): Whether to download the dataset. Overwrites if it already exists
            """
        self.root_dir = os.path.join(root_dir, dataset_name)
        self.mode = sample_mode
        self.train = train
        self.test = test
        self.classes = classes
        self.dataset_name = dataset_name
        # In order to sample in the same way every time
        # and for balancing purposes
        self.cls_counter = {k: 0 for k in self.classes}
        self.balanced = balanced
        self.next_cls = 0

        if download:
            self.save_mnist_to_folders(root_dir)

        self.samples = self._initialize()
        self.length = self.max_samples_length * len(self.classes) if self.balanced else sum([len(v) for v in self.samples.values()])

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

            # create a dictionary for each class
            if current_label not in s:
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
        """Checks if the file should be included in the dataset"""

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
        """
            Returns the sample and target for a given index.

            The function retrieves a sample from the dataset based on the class 
            counters and the current class index. If the dataset is not 'balanced', 
            it uses the provided index to fetch the sample.

            Parameters:
                idx (int): Index of the sample to retrieve.

            Returns:
                tuple: A tuple containing the sample as a 784-dimensional tensor 
                    and the target as a one-hot encoded tensor with 10 dimensions.
        """
        # The idx does not influence the sample, which is returned
        # What is relevant for the sampling is
        # - self.cls_counter
        # - self.next_cls
            
        # first let's get the next class
        cls = self.next_cls
        index = self.update_counters(str(cls))

        # if all samples should be used, we need a different method
        if not self.balanced:
            index = idx
            for key in self.samples:
                current_dict = self.samples[key]
                dict_len = len(current_dict)
                cls = int(key)
                # Check if the index falls within the current dictionary
                if index < dict_len:
                    # Return the idx if within range
                    index = index
                    break
                # Adjust index for the next dictionary
                index -= dict_len

        # now let's get the sample
        x = self.samples[str(cls)][index]

        if self.dataset_name == "cmnist":
            sample = Image.open(x).convert("RGB")
            sample = np.array(sample)
            sample = torch.Tensor(sample).permute(2, 0, 1).squeeze() * 1/255
        else:
            sample = Image.open(x).convert("L")
            sample = np.array([sample])
            sample = torch.Tensor(sample)
            # if self.dataset_name == "fashion_mnist":
            #     sample = sample * 1/255
            # else:
            sample = sample.view(sample.size(0), -1).squeeze() * 1/255
        target = torch.zeros(10)
        target[cls] = 1

        return sample, target

    def save_mnist_to_folders(self, output_dir='data'):
        """
            Saves the dataset to folders on the disk.
            The dataset is saved in a folder structure like this:

            data/dataset_name/
                    train/
                        0/  file1.png
                            file2.png
                            ...
                        1/  ...
                        ...
                    test/
                        0/  ...
                        1/  ...
                        ...
        """
        # Load the MNIST dataset
        if self.dataset_name in ["mnist", "fashion_mnist"]:
            dataset = load_dataset(self.dataset_name)
        elif self.dataset_name == "cmnist":
            dataset = prepare_cmnist_data()
        else:
            raise Exception(f"Dataset '{self.dataset_name}' not supported.")

        output_dir = os.path.join(output_dir, self.dataset_name)
        # Check if it allready exists
        # If so, we assume that the data is already saved
        if os.path.exists(output_dir):
            # print("Data already saved to: ", output_dir)
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
