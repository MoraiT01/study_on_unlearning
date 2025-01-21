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

# torch.manual_seed(100)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

class NoiseGenerator(nn.Module):
    """
    A neural network module for generating noise patterns
    through a series of fully connected layers.
    """

    def __init__(
            self, 
            dim_out: list,
            dim_hidden: list = [1000],
            dim_start: int = 100,
            ):
        """
        Initialize the NoiseGenerator.

        Parameters:
        dim_out (list): The output dimensions for the generated noise.
        dim_hidden (list): The dimensions of hidden layers, defaults to [1000].
        dim_start (int): The initial dimension of random noise, defaults to 100.
        """
        super().__init__()
        self.dim = dim_out
        self.start_dims = dim_start  # Initial dimension of random noise
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Define fully connected layers
        self.layers = {}
        self.layers["l1"] = nn.Linear(self.start_dims, dim_hidden[0]).to(self.device)
        last = dim_hidden[0]
        for idx in range(len(dim_hidden)-1):
            self.layers[f"l{idx+2}"] = nn.Linear(dim_hidden[idx], dim_hidden[idx+1]).to(self.device)
            last = dim_hidden[idx+1]

        # Define output layer
        self.f_out = nn.Linear(last, math.prod(self.dim)).to(self.device)        

    def forward(self):
        """
        Forward pass to transform random noise into structured output.

        Returns:
        torch.Tensor: The reshaped tensor with specified output dimensions.
        """
        # Generate random starting noise
        x = torch.randn(self.start_dims).to(self.device)
        x = x.flatten()

        # Transform noise into learnable patterns
        for layer in self.layers.keys():
            x = self.layers[layer](x)
            x = torch.relu(x)

        # Apply output layer
        x = self.f_out(x)

        # Reshape tensor to the specified dimensions
        reshaped_tensor = x.view(self.dim)
        return reshaped_tensor

def prep_noise_generator(
        forget_data: Dataset,
        model: torch.nn.Module,
        t_Layers: list,
        t_Noise_Dim: int,
    ) -> Tuple[NoiseGenerator, Dict, Dict]:
    """
    Prepares a NoiseGenerator and two dictionaries containing the original labels and the created labels.

    Args:
        forget_data (Dataset): The dataset containing the samples to be forgotten.
        model (torch.nn.Module): The model to be used for generating the created labels.
        t_Layers (list): The dimensions of the hidden layers in the NoiseGenerator.
        t_Noise_Dim (int): The number of features in the generated noise.

    Returns:
        Tuple[NoiseGenerator, Dict, Dict]: A tuple containing the NoiseGenerator, the original labels and the created labels.
    """
    
    # Create two dictionaries to store the original labels and the created labels
    noises = {}
    og_labels = {}
    created_labels =  {}

    # Set the model to evaluation mode
    model.eval()

    # Iterate over the forget_data and create the created labels
    for index, data in enumerate(DataLoader(forget_data, batch_size=1, shuffle=False)):
        # Get the sample and the label
        s, l = data

        # Use the model to generate the created label
        new_l = F.softmax(model(s.to(DEVICE)).detach(), dim=1)
        
        # Store the created label and the original label
        created_labels[index] = new_l[0].to(DEVICE)
        og_labels[index] = l[0].to(DEVICE)

    # Create a NoiseGenerator with the specified dimensions
    noises = NoiseGenerator(s[0].shape, t_Layers, t_Noise_Dim).to(DEVICE)

    # Return the NoiseGenerator and the two dictionaries
    return noises, created_labels, og_labels

class NoiseDataset(Dataset):
    """
    A DataLoader which uses a noise generator to generate data and labels.

    Args:
        noise_generator (NoiseGenerator): The noise generator to use.
        noise_labels (Dict[int, torch.Tensor] | torch.Tensor): The labels to use. If a tensor, it is used as the labels for all samples. If a dict, it is used as a mapping of indices to labels.
        number_of_noise (int, optional): The number of noise samples to generate. Defaults to 100.
    """

    def __init__(self, noise_generator: NoiseGenerator, noise_labels: Dict[int, torch.Tensor] | torch.Tensor, number_of_noise: int = 100,):

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
        return self.noise_generator().to(DEVICE), self.noise_labels.to(DEVICE) if isinstance(self.noise_labels, torch.Tensor) else self.noise_labels[idx]

def noise_maximization(
        forget_data: Dataset,
        model: torch.nn.Module,
        t_Epochs: int,
        t_Learning_Rate: float,
        t_Batch_Size: int,
        t_Regularization_term: float,
        t_Layers: list,
        t_Noise_Dim: int,
        logs: bool = False,) -> Tuple[NoiseGenerator, Dict[int, torch.Tensor] | torch.Tensor]:
    """
    This function maximizes the loss of the model on the forget_data by generating noise with the noise generator
    and adding it to the dataset.

    Args:
        forget_data (Dataset): The dataset to use for the unlearning process.
        model (torch.nn.Module): The model to be unlearned.
        t_Epochs (int): The number of epochs to train the noise generator.
        t_Learning_Rate (float): The learning rate of the noise generator.
        t_Batch_Size (int): The batch size of the DataLoader.
        t_Regularization_term (float): The regularization term to add to the loss.
        t_Layers (list): The layers of the noise generator.
        t_Noise_Dim (int): The dimension of the noise to generate.
        logs (bool, optional): Whether to print logs. Defaults to False.

    Returns:
        Tuple[NoiseGenerator, Dict[int, torch.Tensor] | torch.Tensor]: A tuple containing the noise generator and the labels of the forget_data.
    """
    noise_generator, _, og_labels = prep_noise_generator(
        forget_data,
        model,
        t_Layers=t_Layers,
        t_Noise_Dim=t_Noise_Dim,
    )
    noise_loader = DataLoader(
        dataset=NoiseDataset(noise_generator, og_labels),
        batch_size=t_Batch_Size, # Hyperparameter
        shuffle=True,
    )

    model.to(DEVICE)
    model.eval()
    optimizers = torch.optim.Adam(noise_generator.parameters(), lr = t_Learning_Rate) # Hyperparameter
    num_epochs = t_Epochs # Hyperparameter

    d = [1] if forget_data.dataset_name in ["mnist", "fashion_mnist"]  else [1,2,3]
    epoch = 0
    while True:
        total_loss = []
        epoch += 1
        for input_batch, l in noise_loader:

            outputs = model(input_batch)
            loss = - F.cross_entropy(outputs, l) + t_Regularization_term * torch.mean(torch.sum(torch.square(input_batch), d))
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()

            # for logging
            total_loss.append(loss.cpu().detach().numpy())

        # scheduler.step()
        if logs:
            print("Epoch: {}, Loss: {}".format(epoch, np.mean(total_loss)))

        if epoch >= num_epochs:
            # Does the loss have to be less than 0?
            break
    
    return noise_generator, og_labels


class FeatureMU_Loader(Dataset):
    """
    This class creates a new dataset which contains the given retain_data and some noise generated by the noise_generator.
    The noise is generated with the label4noise and added to the dataset as many times as specified in number_of_noise.
    """

    def __init__(self, noise_generator: NoiseGenerator, label4noise: Dict[int, torch.Tensor] | torch.Tensor, number_of_noise: int, retain_data: Dataset):
        """
        Initializes the FeatureMU_Loader, containing a generator for noise and a dataset which contains the data that should be retained.

        Args:
            noise_generator (NoiseGenerator): The noise generator which generates the noise.
            label4noise (torch.Tensor): The label which is used to generate the noise.
            number_of_noise (int): The number of noise samples to be generated.
            retain_data (Dataset): The dataset which contains the data that should be retained.
        """
        self.noise_gen = noise_generator
        self.retain_data = retain_data
        self.number_of_noise = number_of_noise
        self.label4noise = label4noise

        # Set the noise generator to evaluation mode
        self.noise_gen.eval()

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        This includes the number of noise samples and the number of samples
        in the retained data.

        Returns:
            int: The total number of samples in the dataset.
        """
        # Calculate the total number of samples by summing the number of noise
        # samples and the number of samples in the retained dataset.
        return len(self.retain_data) + self.number_of_noise

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a sample from the dataset.

        If the index is less than the number of noise samples, it returns a noise sample and the label.
        Otherwise, it returns a sample from the retained data.

        Args:
            idx (int): The index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sample and the label.
        """
        if idx < self.number_of_noise:
            # Get the label from the label4noise dictionary if it's a dictionary
            label = self.label4noise if isinstance(self.label4noise, torch.Tensor) else self.label4noise[idx]
            # Return the noise sample and the label
            return self.noise_gen().to(DEVICE), label.to(DEVICE)
        else:
            # Return a sample from the retained data
            s, l = self.retain_data.__getitem__(idx - self.number_of_noise)
            return s.to(DEVICE), l.to(DEVICE)

def impairing_phase(noise_generator: NoiseGenerator, number_of_noise: int, label4noise: torch.Tensor, retain_data: Dataset, t_Impair_LR: float, model: torch.nn.Module, logs: bool = False) -> torch.nn.Module:
    """
    The impairing phase of the MU algorithm inspired by https://github.com/vikram2000b/Fast-Machine-Unlearning
    In this phase, the model is trained with the data from the retain_loader and some noise generated by the noise_generator.
    The noise is generated with the label4noise and added to the dataset as many times as specified in number_of_noise.

    Args:
        noise_generator (NoiseGenerator): The noise generator which generates the noise.
        number_of_noise (int): The number of noise samples to be generated.
        label4noise (torch.Tensor): The label which is used to generate the noise.
        retain_loader (Dataset): The dataset which contains the data that should be retained.
        model (torch.nn.Module): The model to be trained.
        logs (bool, optional): Whether to print logs. Defaults to False.

    Returns:
        torch.nn.Module: The model after the impairing phase.

    """
    noisy_loader = DataLoader(
        dataset=FeatureMU_Loader(noise_generator, label4noise, number_of_noise, retain_data),
        batch_size=8, 
        shuffle=True
    )
    optimizer = torch.optim.Adam(model.parameters(), lr = t_Impair_LR) # Hyperparameter

    for epoch in range(1): # Hyperparameter?
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

def repairing_phase(retain_data: Dataset, model: torch.nn.Module, t_Repair_LR: float, logs: bool = False) -> torch.nn.Module:
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

    optimizer = torch.optim.Adam(model.parameters(), lr = t_Repair_LR) # Hyperparameter

    for epoch in range(1): # Hyperparameter?
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
        t_Epochs: int = 9,
        t_Learning_Rate: float = 0.03,
        t_Batch_Size: int = 256, 
        t_N2R_Ratio: int = 1.9,
        t_Regularization_term: float = 0.15,
        t_Layers: list = [1024,1024,1024,1024,1024,1024,1024,1024,1024,],
        t_Noise_Dim: int = 126,
        t_Impair_LR: float = 0.02,
        t_Repair_LR: float = 0.02,
        logs: bool = False,
        model_eval_logs: bool = False,
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
    torch.cuda.empty_cache()
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
    if model_eval_logs:
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
        print("Starting NoiseGenerator Maximazation Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))
    
    # Get the Noise
    noise_gen, noise_labels = noise_maximization(
        forget_data=data_forget,
        model=model,
        t_Epochs=t_Epochs,
        t_Learning_Rate=t_Learning_Rate,
        t_Batch_Size=t_Batch_Size,
        t_Regularization_term=t_Regularization_term,
        t_Layers=t_Layers,
        t_Noise_Dim=t_Noise_Dim,
        logs=logs,
    )

    if logs:
        print("Ending NoiseGenerator Maximazation Phase")
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
    retain_data.length = int(len(data_forget) * t_N2R_Ratio) # every except the one we want to forget from
    if logs:
        print("______")
        print("Starting Impairing Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))
    
    impaired_model = impairing_phase(
        noise_generator=noise_gen,
        number_of_noise=len(data_forget),
        label4noise=noise_labels,
        retain_data=retain_data,
        model=model,
        t_Impair_LR=t_Impair_LR,
        logs=logs,
    )

    if logs:
        print("Ending Impairing Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))
    if model_eval_logs:
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
        t_Repair_LR=t_Repair_LR,
        logs=logs,
    )

    if logs:
        print("Ending Repairing Phase")
        print("Time: {}".format(datetime.datetime.now().timestamp() - start_time))
        print("______")
    if model_eval_logs:
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
