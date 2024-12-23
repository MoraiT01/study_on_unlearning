"""
    contains all the code necessary for running the MU algorithm inspired by https://github.com/vikram2000b/Fast-Machine-Unlearning
    Instead of unlearning one entire class, we focus on unlearning a subset of on class, grouped together by one shared feature
        -> hence: Feature Unlearning
"""

# import required libraries
import numpy as np

from typing import Literal, Tuple, Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# my
from training import model_params
from mlp_dataclass import MNIST_CostumDataset
from my_random import shared_random_state
import math
import datetime
from tqdm import tqdm

torch.manual_seed(100)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOISE_BATCH_SIZE = 256

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    _, l = torch.max(labels, dim=1)
    correct = torch.sum(preds == l).item()
    return torch.tensor(correct / len(preds))

def validation_step(model, batch):
    images, labels = batch
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    out = model(images)                    
    loss = F.cross_entropy(out, labels)   
    acc = accuracy(out, labels)
    return {'Loss': loss.detach(), 'Acc': acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)

class NoiseGen(nn.Module):
    """
    A neural network module for generating noise with a specified dimension.
    """

    def __init__(self, dim):
        """
        Initializes the NoiseGen module.

        Args:
            dim (tuple): The dimensions of the output noise.
        """
        super().__init__()
        self.dim = dim
        self.start_dims = 100  # Initial dimension of random noise

        # Define fully connected layers
        self.f1 = nn.Linear(self.start_dims, 1000)
        self.f2 = nn.Linear(1000, math.prod(self.dim))

    def forward(self):
        """
        Performs a forward pass to generate noise.

        Returns:
            torch.Tensor: A tensor reshaped to the specified dimensions.
        """
        # Generate random starting noise
        x = torch.randn(self.start_dims)
        x = x.flatten()

        # Transform noise into learnable patterns
        x = self.f1(x)
        x = torch.relu(x)
        x = self.f2(x)

        # Reshape tensor to the specified dimensions
        reshaped_tensor = x.view(self.dim)
        return reshaped_tensor

def prep_noise_generator(forget_data: Dataset, model: torch.nn.Module) -> Tuple[NoiseGen, Dict, Dict]:
    """
    Creates a noise generator to generate noise which mimics the data in forget_data.

    Args:
        forget_data (Dataset): The dataset which contains the data to be forgotten.
        model (torch.nn.Module): The model which was used to generate the labels.

    Returns:
        Tuple[NoiseGen, Dict, Dict]:
            - The trained NoiseGen model.
            - A dictionary which contains the original labels.
            - A dictionary which contains the new labels.
    """
    
    noises = {}
    og_labels = {}
    created_labels =  {}
    model.eval()
    for index, data in enumerate(DataLoader(forget_data, batch_size=1, shuffle=False)): # iterate over forget_data):
        s, l = data

        new_l = F.softmax(model(s.to(DEVICE)).detach(), dim=1)
        
        created_labels[index] = new_l[0].to(DEVICE)
        og_labels[index] = l[0].to(DEVICE)
        # Ursprünglich waren hier die Labels der der Klassen gemeint
        # Jedoch entschied ich mich dagegen
        # Der prognostizierte Wahrkeitsvektor ist eine andere Darstellung des Samples,
        # Wir wollen nicht die Klasse unlearnen, sonder das Sample/das Feature

    noises = NoiseGen(s[0].shape).to(DEVICE)

    return noises, created_labels, og_labels

def prep_noise_generator(forget_data: Dataset, model: torch.nn.Module) -> Tuple[NoiseGen, Dict, Dict]:
    """
    Creates a noise generator to generate noise which mimics the data in forget_data.

    Args:
        forget_data (Dataset): The dataset which contains the data to be forgotten.
        model (torch.nn.Module): The model which was used to generate the labels.

    Returns:
        Tuple[NoiseGen, Dict, Dict]:
            - The trained NoiseGen model.
            - A dictionary which contains the original labels.
            - A dictionary which contains the new labels.
    """
    
    noises = {}
    og_labels = {}
    created_labels =  {}
    model.eval()
    for index, data in enumerate(DataLoader(forget_data, batch_size=1, shuffle=False)): # iterate over forget_data):
        s, l = data

        new_l = F.softmax(model(s.to(DEVICE)).detach(), dim=1)
        
        created_labels[index] = new_l[0].to(DEVICE)
        og_labels[index] = l[0].to(DEVICE)
        # Ursprünglich waren hier die Labels der der Klassen gemeint
        # Jedoch entschied ich mich dagegen
        # Der prognostizierte Wahrkeitsvektor ist eine andere Darstellung des Samples,
        # Wir wollen nicht die Klasse unlearnen, sonder das Sample/das Feature

    noises = NoiseGen(s[0].shape).to(DEVICE)

    return noises, created_labels, og_labels

class NoiseDataset(Dataset):
    """
    A DataLoader which uses a noise generator to generate data and labels.

    Args:
        noise_generator (NoiseGen): The noise generator to use.
        noise_labels (Dict[int, torch.Tensor] | torch.Tensor): The labels to use. If a tensor, it is used as the labels for all samples. If a dict, it is used as a mapping of indices to labels.
        number_of_noise (int, optional): The number of noise samples to generate. Defaults to 100.
    """

    def __init__(self, noise_generator: NoiseGen, noise_labels: Dict[int, torch.Tensor] | torch.Tensor, number_of_noise: int = 100,):

        self.noise_generator = noise_generator
        self.noise_labels  = noise_labels
        self.number_of_noise = number_of_noise if isinstance(self.noise_labels, torch.Tensor) else len(self.noise_labels)

    def __len__(self) -> int:
        """
        The number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.number_of_noise
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sample and the label.
        """
        return self.noise_generator(), self.noise_labels if isinstance(self.noise_labels, torch.Tensor) else self.noise_labels[idx]

def noise_maximization(forget_data: Dataset, model: torch.nn.Module, logs: bool = False) -> Tuple[NoiseGen, Dict[int, torch.Tensor] | torch.Tensor]:
    """
    This function trains a noise generator to maximize the output of a model on given labels.

    Args:
        forget_data (Dataset): The dataset which contains the data to be forgotten.
        model (torch.nn.Module): The model which was used to generate the labels.
        logs (bool): Whether to print logs.

    Returns:
        Tuple[NoiseGen, torch.Tensor]:
            - The trained NoiseGen model.
            - The mean of the labels which were used to train the noise generator.
    """
    noise_generator, _, og_labels = prep_noise_generator(forget_data, model)
    noise_loader = DataLoader(
        dataset=NoiseDataset(noise_generator, og_labels),
        batch_size=32, # Hyperparameter
        shuffle=True,
    )

    model.to(DEVICE)
    model.eval()
    optimizers = torch.optim.Adam(noise_generator.parameters(), lr = 0.02) # Hyperparameter
    num_epochs = 5 # Hyperparameter
    # Optional learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizers, step_size=int(num_epochs/3), gamma=0.1)
    d = [1] if forget_data.dataset_name in ["mnist", "fashion_mnist"]  else [1,2,3]

    epoch = 0
    while True:
        total_loss = []
        epoch += 1
        for input_batch, l in noise_loader:

            outputs = model(input_batch)
            loss = - F.cross_entropy(outputs, l) + 0.1 * torch.mean(torch.sum(torch.square(input_batch), d))
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()

            # for logging
            total_loss.append(loss.cpu().detach().numpy())

        # scheduler.step()
        if logs:
            print("Epoch: {}, Loss: {}".format(epoch, np.mean(total_loss)))

        if epoch >= num_epochs:
            # the los needs to be below 0
            # not a very elegant solution, but needed
            if loss < 0:
                break
            else:
                # give it a few more epochs to train
                # this is meant to be a sort of failsave
                num_epochs += 1
        
    # created_labels = torch.stack(list(created_labels.values()))
    # created_label = torch.mean(created_labels, dim=0)
    
    return noise_generator, og_labels


class FeatureMU_Loader(Dataset):
    """
    This class creates a new dataset which contains the given retain_data and some noise generated by the noise_generator.
    The noise is generated with the label4noise and added to the dataset as many times as specified in number_of_noise.
    """

    def __init__(self, noise_generator: NoiseGen, label4noise: Dict[int, torch.Tensor] | torch.Tensor, number_of_noise: int, retain_data: Dataset):
        """

        """
        self.noise_gen = noise_generator
        self.retain_data = retain_data
        self.number_of_noise = number_of_noise
        self.label4noise = label4noise

        self.noise_gen.eval()

    def __len__(self) -> int:
        """

        """
        return len(self.retain_data) + self.number_of_noise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        """
        if idx < self.number_of_noise:
            label = self.label4noise if isinstance(self.label4noise, torch.Tensor) else self.label4noise[idx]
            return self.noise_gen()[idx], label
        else:
            return self.retain_data.__getitem__(idx - self.number_of_noise)

def impairing_phase(noise_batch: NoiseGen, number_of_noise: int, label4noise: torch.Tensor, retain_data: Dataset, model: torch.nn.Module, logs: bool = False) -> torch.nn.Module:
    """

    """
    noisy_loader = DataLoader(
        dataset=FeatureMU_Loader(noise_batch, label4noise, number_of_noise, retain_data),
        batch_size=8, 
        shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.02) # Hyperparameter

    for epoch in range(1): # Hyperparameter  
        model.train(True)
        running_loss = 0.0
        
        for i, data in enumerate(noisy_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.clone().detach().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # Append the loss to the list of losses
            running_loss += loss.item() * inputs.size(0)

        if logs:
            print(f"Train loss {epoch+1}: {running_loss/len(noisy_loader)}")

    return model

def repairing_phase(retain_data: Dataset, model: torch.nn.Module, logs: bool = False) -> torch.nn.Module:
    """
    Perform the repairing phase of the Fast Machine Unlearning algorithm.

    Parameters:
        retain_data (Dataset): The retained data.
        model (torch.nn.Module): The model to be repaired.
        logs (bool, optional): Whether to print logs. Defaults to False.

    Returns:
        torch.nn.Module: The model after the repairing phase.
    """
    heal_loader = torch.utils.data.DataLoader(retain_data, batch_size=8, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) # Hyperparameter

    for epoch in range(1): # Hyperparameter
        model.train(True)
        running_loss = 0.0
        
        for i, data in enumerate(heal_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.clone().detach().to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Append the loss to the list of losses
            running_loss += loss.item() * inputs.size(0)

        if logs:
            print(f"Train loss {epoch+1}: {running_loss/len(heal_loader)}")

    return model

def _main(
        model: torch.nn.Module,
        dataset_name: Literal["mnist", "cmnist", "fashion_mnist"],
        logs: bool = False,
    ) -> torch.nn.Module:
    """
    Main function for the Fast Machine Unlearning inspired algorithm.

    Parameters:
        model (torch.nn.Module): The model to be unlearned.
        dataset_name (Literal["mnist", "cmnist", "fashion_mnist"]): The name of the dataset to use.
        logs (bool, optional): Whether to print logs. Defaults to False.

    Returns:
        torch.nn.Module: The model after the unlearning process.
    """
    model.to(DEVICE)
    start_time = datetime.datetime.now().timestamp()

    # Validation Dataloaders for the forget data
    forget_valid_dl = DataLoader(
        dataset = MNIST_CostumDataset(
            sample_mode="only_erased",
            train=False,
            test=True,
            dataset_name=dataset_name,
        ),
        batch_size=8,
        shuffle=False,
    )
    # Validation Dataloader for the retained data
    retain_valid_dl = DataLoader(
        dataset = MNIST_CostumDataset(
            sample_mode="except_erased",
            train=False,
            test=True,
            dataset_name=dataset_name,
        ),
        batch_size=8,
        shuffle=False,
    )
    # the forget train data
    data_forget = MNIST_CostumDataset(
        sample_mode="only_erased",
        train=True,
        test=False,
        dataset_name=dataset_name,
    )
    
    if logs:
        print("Baseline Performance")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))

        print("______")
        print("Performance of Baseline on Forget Class")
        history = [evaluate(model, forget_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("Performance of Baseline Model on Retain Class")
        history = [evaluate(model, retain_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("______")
        print("Starting NoiseGen Maximazation Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))
    
#   # Get the Noise
    noise_gen, noise_labels = noise_maximization(
        forget_data=data_forget,
        model=model,
        logs=logs,
    )

    if logs:
        print("Ending NoiseGen Maximazation Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))

    # the retain train data
    retain_data = MNIST_CostumDataset(
            sample_mode="except_erased",
            train=True,
            test=False,
            balanced=True,
            dataset_name=dataset_name,
        )
    
    # We need to make sure that the cls are balanced
    # take the same amout like in the paper of femu
    # 1000 samples per other class
    retain_data.length = 1000 * 9 # every except the one we want to forget from
    if logs:
        print("______")
        print("Starting Impairing Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))
    
    impaired_model = impairing_phase(
        noise_generator=noise_gen,
        number_of_noise=NOISE_BATCH_SIZE,
        label4noise=noise_labels,
        retain_data=retain_data,
        model=model,
        logs=logs,
    )

    if logs:
        print("Ending Impairing Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))

        print("______")
        print("Performance of Impaired Model on Forget Class")
        history = [evaluate(impaired_model, forget_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("Performance of Impaired Model on Retain Class")
        history = [evaluate(impaired_model, retain_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("______")
        print("Starting Repairing Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))
    
    repaired_model = repairing_phase(
        retain_data=retain_data,
        model=impaired_model,
        logs=logs,
    )

    if logs:
        print("Ending Repairing Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))
        print("______")

        print("Performance of repaired Model on Forget Class")
        history = [evaluate(repaired_model, forget_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("Performance of Repaired Model on Retain Class")
        history = [evaluate(repaired_model, retain_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))

    return repaired_model

