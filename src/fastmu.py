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

torch.manual_seed(100)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

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

def train_noise_generator(forget_data: Dataset, model: torch.nn.Module) -> Tuple[NoiseGen, Dict, Dict]:
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
        og_labels[index] = new_l.to(DEVICE)
        # Ursprünglich waren hier die Labels der der Klassen gemeint
        # Jedoch entschied ich mich dagegen
        # Der prognostizierte Wahrkeitsvektor ist eine andere Darstellung des Samples,
        # Wir wollen nicht die Klasse unlearnen, sonder das Sample/das Feature

    noises = NoiseGen(s[0].shape).to(DEVICE)

    return noises, created_labels, og_labels

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
    noise_generator, created_labels, _ = train_noise_generator(forget_data, model)
    created_l_loader = DataLoader(
        created_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model.to(DEVICE)
    optimizers = torch.optim.Adam(noise_generator.parameters(), lr = 0.1) # Hyperparameter
    num_epochs = 100 # Hyperparameter
    # Optional learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizers, step_size=int(num_epochs/5), gamma=0.1)

    for epoch in range(num_epochs):
        total_loss = []

        for l in created_l_loader:

            # Firstly, generate the noise
            input_batch = None
            for i in range(len(l)):	# My Standard Batchsize
                new = noise_generator().to(DEVICE).unsqueeze(0)
                input_batch = new if input_batch == None else torch.cat((input_batch, new), dim=0)

            outputs = model(input_batch)
            loss = - F.cross_entropy(outputs, l) + 0.1 * torch.mean(torch.sum(torch.square(input_batch)))
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()

            # for logging
            total_loss.append(loss.cpu().detach().numpy())

        # scheduler.step()
        if (epoch+1)%5 == 0 and logs:    
            print("Epoch: {}, Loss: {}".format(epoch, np.mean(total_loss)))

    # created_labels = torch.stack(list(created_labels.values()))
    # created_label = torch.mean(created_labels, dim=0)
    
    return noise_generator, created_labels

class FeatureMU_Loader(Dataset):
    """
    This class creates a new dataset which contains the given retain_data and some noise generated by the noise_generator.
    The noise is generated with the label4noise and added to the dataset as many times as specified in number_of_noise.
    """

    def __init__(self, noise_generator: NoiseGen, label4noise: Dict[int, torch.Tensor] | torch.Tensor, number_of_noise: int, retain_data: Dataset):
        """
        Initializes the FeatureMU_Loader, containing a generator for noise and a dataset which contains the data that should be retained.

        Args:
            noise_generator (NoiseGen): The noise generator which generates the noise.
            label4noise (torch.Tensor): The label which is used to generate the noise.
            number_of_noise (int): The number of noise samples to be generated.
            retain_data (Dataset): The dataset which contains the data that should be retained.
        """
        self.noise_gen = noise_generator
        self.retain_data = retain_data
        self.number_of_noise = number_of_noise
        self.label4noise = label4noise

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.retain_data) + self.number_of_noise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sample from the dataset.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sample and the label.
        """
        if idx < self.number_of_noise:
            label = self.label4noise if isinstance(self.label4noise, torch.Tensor) else self.label4noise[idx]
            return self.noise_gen(), label
        else:
            return self.retain_data.__getitem__(idx - self.n)

def impairing_phase(noise_generator: NoiseGen, number_of_noise: int, label4noise: torch.Tensor, retain_loader: Dataset, model: torch.nn.Module, logs: bool = False) -> torch.nn.Module:
    """
    The impairing phase of the MU algorithm inspired by https://github.com/vikram2000b/Fast-Machine-Unlearning
    In this phase, the model is trained with the data from the retain_loader and some noise generated by the noise_generator.
    The noise is generated with the label4noise and added to the dataset as many times as specified in number_of_noise.

    Args:
        noise_generator (NoiseGen): The noise generator which generates the noise.
        number_of_noise (int): The number of noise samples to be generated.
        label4noise (torch.Tensor): The label which is used to generate the noise.
        retain_loader (Dataset): The dataset which contains the data that should be retained.
        model (torch.nn.Module): The model to be trained.
        logs (bool, optional): Whether to print logs. Defaults to False.

    Returns:
        torch.nn.Module: The model after the impairing phase.
    """
    noisy_loader = DataLoader(
        dataset=FeatureMU_Loader(noise_generator, label4noise, number_of_noise, retain_loader),
        batch_size=8, 
        shuffle=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.02) # Hyperparameter

    for epoch in range(1): # Hyperparameter  
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(noisy_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), torch.tensor(labels).cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(),dim=1)
            assert out.shape==labels.shape
            running_acc += (labels==out).sum().item()
        if logs:
            print(f"Train loss {epoch+1}: {running_loss/len(noisy_loader)}, Train Acc:{running_acc*100/len(noisy_loader)}%")

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
        running_acc = 0
        for i, data in enumerate(heal_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), torch.tensor(labels).cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(),dim=1)
            assert out.shape==labels.shape
            running_acc += (labels==out).sum().item()
        print(f"Train loss {epoch+1}: {running_loss/len(retain_data)},Train Acc:{running_acc*100/len(retain_data)}%")

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

    # (Re)Training Dataloaders
    noise_gen, noise_labels = noise_maximization(
        forget_data=data_forget,
        model=model,
        logs=logs,
    )

    # the retain train data
    retain_data = MNIST_CostumDataset(
            sample_mode="except_erased",
            train=True,
            test=False,
            balanced=True,
            dataset_name=dataset_name,
        )
    
    # We need to make sure that the cls are balanced
    retain_data.length = len(data_forget) * len(retain_data.classes)
    
    impaired_model = impairing_phase(
        noise_generator=noise_gen,
        number_of_noise=len(data_forget),
        label4noise=noise_labels,
        retain_data=retain_data,
        model=model,
        logs=logs,
    )

    if logs:
        print("Performance of Standard Forget Model on Forget Class")
        history = [evaluate(impaired_model, forget_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("Performance of Standard Forget Model on Retain Class")
        history = [evaluate(impaired_model, retain_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))


    repaired_model = repairing_phase(
        retain_data=retain_data,
        model=impaired_model,
        logs=logs,
    )

    if logs:
        print("Performance of Standard Forget Model on Forget Class")
        history = [evaluate(repaired_model, forget_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("Performance of Standard Forget Model on Retain Class")
        history = [evaluate(repaired_model, retain_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

    return repaired_model