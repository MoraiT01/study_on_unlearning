"""
    This file contains unlearning algorithms
"""
import torch
import numpy as np
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from typing import Literal, Tuple, Dict
from copy import deepcopy
from tqdm import tqdm

from training import model_params

from abc import ABC, abstractmethod
# Set device (use GPU if available)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Unlearner(ABC):

    @abstractmethod
    def __init__(self, model: torch.nn.Module, unlearned_data: DataLoader) -> None:
        pass

    @abstractmethod
    def unlearn(self) -> torch.nn.Module:
        pass

# First we should try my most intuative unlearning algorithm
# -> Gradient Ascent on forget Samples

class SimpleGradientAscent(Unlearner):

    def __init__(self, model: torch.nn.Module, unlearned_data: DataLoader, dataset_name: Literal["mnist", "cmnist", "fashion_mnist"]) -> None:
        self.model = model
        self.unlearned_data = unlearned_data

        self._initialize(dataset_name=dataset_name)

    def _initialize(self, dataset_name) -> None:
        """Initializes the unlearning algorithm hyperparameters"""

        self.lr = model_params[dataset_name]["lr"]
        # self.n_updates = model_params[dataset_name]["n_updates"]
        # self.eval_steps = model_params[dataset_name]["eval_steps"]

    def __str__(self) -> str:
        return "SimpleGradientAscent"
    
    def unlearn(self,) -> torch.nn.Module:
        """Unlearns the model; Copys the model first, unlearns and return the new model"""
        new_model = deepcopy(self.model)
        new_model.to(DEVICE)

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.lr, maximize=True)

        for images, labels in tqdm(self.unlearned_data, desc="Unlearning samples", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Zero the gradients
            new_model.zero_grad()

            # Forward pass
            outputs = new_model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

        return new_model

class FastEffiecentUnlearning(Unlearner):

    def __init__(
            self,
            model: torch.nn.Module,
            unlearned_data: DataLoader,
            complete_data: DataLoader,
            dataset_name: Literal["mnist", "cmnist", "fashion_mnist"],
            ) -> None:
        self.model = model
        self.data = unlearned_data
        self.complete_data = complete_data
        self.lr = model_params[dataset_name]["lr"]
        self.n_updates = model_params[dataset_name]["n_updates"]

        self._initialize(dataset_name=dataset_name)

    def _initialize(self, dataset_name) -> None:
        pass

    def __str__(self) -> str:
        return "FastEffiecentUnlearning"
    
    def unlearn(self) -> Module:
        new_model = deepcopy(self.model)
        new_model.to(DEVICE)


        loss_function = torch.nn.CrossEntropyLoss()
        # named opt_func in the original code
        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.lr, maximize=True)
        return 

def get_unlearners(name: Literal["SimpleGradientAscent", "FastEffiecentUnlearning"], dataset_name: Literal["mnist", "cmnist", "fashion_mnist"], args: dict = None) -> Dict[str, Unlearner]:
    if name == "SimpleGradientAscent":
        return {k: SimpleGradientAscent(v, args["u_data"], dataset_name=dataset_name) for k, v in args["models"].items()}
    elif name == "FastEffiecentUnlearning":
        return {k: FastEffiecentUnlearning(v, args["u_data"], args["data"], dataset_name=dataset_name) for k, v in args["models"].items()}
    else:
        raise Exception(f"Unlearning algorithm '{name}' not supported.")

def unlearn_n_models(
        models: Dict[str, torch.nn.Module],
        unlearned_data: DataLoader,
        dataset_name: Literal["mnist", "cmnist", "fashion_mnist"],
        which_unlearning: Literal["SimpleGradientAscent", "FastEffiecentUnlearning"],
        data: DataLoader = None,
        logs: bool = True,
        ) -> Dict[str, torch.nn.Module]:
    
    unlearners = get_unlearners(
        name=which_unlearning,
        dataset_name=dataset_name,
        args={"models": models, "u_data": unlearned_data, "data": data}
    )
    
    new_models = {}
    if logs:
        print(f"Unlearning {len(unlearners)} models trained on {dataset_name}")
    for k, model in unlearners.items():
        new_models[k] = model.unlearn()
        if logs:
            print(f"Unlearned model {k:2}/{len(unlearners):2}...")

    return new_models
