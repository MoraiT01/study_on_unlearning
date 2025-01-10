"""
    This file contains unlearning algorithms
"""
import torch
from torch.nn.modules import Module
from torch.utils.data import DataLoader
from typing import Literal, Dict
from copy import deepcopy
from tqdm import tqdm

from training import model_params
import gefeu
import gemu

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
    """
    The Simple Gradient Ascent unlearning algorithm.

    This algorithm is the most intuitive unlearning algorithm. It simply performs gradient ascent on the samples which should be unlearned.
    """
    def __init__(
            self,
            model: torch.nn.Module,
            unlearned_data: DataLoader,
            dataset_name: Literal["mnist", "cmnist", "fashion_mnist"],
            t_LR: float = None,
            t_Epochs: int = 1,
            ) -> None:
        """
        Initializes the SimpleGradientAscent unlearning algorithm.

        Parameters:
            model (torch.nn.Module): The model to be unlearned.
            unlearned_data (DataLoader): The data which should be unlearned.
            dataset_name (Literal["mnist", "cmnist", "fashion_mnist"]): The name of the dataset.
        """
        self.model = model
        self.unlearned_data = unlearned_data

        self._initialize(dataset_name=dataset_name)
        if t_LR is not None:
            self.lr = t_LR
        self.epochs = t_Epochs

    def _initialize(self, dataset_name) -> None:
        """Initializes the unlearning algorithm hyperparameters"""

        self.lr = model_params[dataset_name]["lr"]
        # self.n_updates = model_params[dataset_name]["n_updates"]
        # self.eval_steps = model_params[dataset_name]["eval_steps"]

    def __str__(self) -> str:
        """Returns a string representation of the unlearning algorithm"""
        return "SimpleGradientAscent"
    
    def unlearn(self,) -> torch.nn.Module:
        """
        Unlearns the model; Copys the model first, unlearns and return the new model

        Returns:
            torch.nn.Module: The new model after the unlearning process.
        """
        new_model = deepcopy(self.model)
        new_model.to(DEVICE)

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=self.lr, maximize=True)

        for epoch in range(self.epochs):
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

class GeneratorMachineUnlearning(Unlearner):
    """
    This algorithm is inspired by the paper "Fast Machine Unlearning" by Vikram et al.
    My Goal: The task is to unlearn a subset of a class, not the whole class. The subset should be tide together be one shared feature.
    Core Difference: 
        - Instead of a noise batch we use a noise Generator
        - D_Retain and D_forget are change to fit the new task 
    """

    def __init__(
            self,
            model: torch.nn.Module,
            dataset_name: Literal["mnist", "cmnist", "fashion_mnist"],
            ) -> None:
        """
        Initializes the GeneratorMachineUnlearning algorithm.

        Parameters:
            model (torch.nn.Module): The model to be unlearned.
            dataset_name (Literal["mnist", "cmnist", "fashion_mnist"]): The name of the dataset.
        """
        self.model = model
        self.dataset_name = dataset_name

    def __str__(self) -> str:
        return "GeneratorMachineUnlearning"
    
    def unlearn(self, logs: bool = False) -> Module:
        """
        Unlearns the model according to the GeneratorMachineUnlearning algorithm.

        Returns:
            Module: The model after unlearning.
        """
        new_model = deepcopy(self.model)
        new_model.to(DEVICE)

        new_model = gemu._main(
            model=new_model,
            dataset_name=self.dataset_name,
            logs=logs,
            )

        return new_model

class GeneratorFeatureUnlearning(Unlearner):
    """
    This algorithm is inspired by the paper "Fast Machine Unlearning" by Vikram et al.
    My Goal: The task is to unlearn a subset of a class, not the whole class. The subset should be tide together be one shared feature.
    Core Difference: 
        - Instead of a noise batch we use a noise Generator
        - D_Retain and D_forget are change to fit the new task
        - Noise Generator is trained to maximize the Prior Probabilities predicted by the model, instead of the class label.
    """

    def __init__(
            self,
            model: torch.nn.Module,
            dataset_name: Literal["mnist", "cmnist", "fashion_mnist"],
            ) -> None:
        """
        Initializes the GeneratorFeatureUnlearning algorithm.

        Parameters:
            model (torch.nn.Module): The model to be unlearned.
            dataset_name (Literal["mnist", "cmnist", "fashion_mnist"]): The name of the dataset.
        """
        self.model = model
        self.dataset_name = dataset_name

    def __str__(self) -> str:
        return "GeneratorFeatureUnlearning"
    
    def unlearn(self, logs: bool = False) -> Module:
        """
        Unlearns the model according to the GeneratorFeatureUnlearning algorithm.

        Returns:
            Module: The model after unlearning.
        """
        new_model = deepcopy(self.model)
        new_model.to(DEVICE)

        new_model = gefeu._main(
            model=new_model,
            dataset_name=self.dataset_name,
            logs=logs,
            )

        return new_model

def get_unlearners(name: Literal["SimpleGradientAscent", "GeneratorFeatureUnlearning", "GeneratorMachineUnlearning"], dataset_name: Literal["mnist", "cmnist", "fashion_mnist"], args: dict = None) -> Dict[str, Unlearner]:
    """
    Returns a dictionary of Unlearners for all models in args["models"].

    Parameters:
        name (Literal["SimpleGradientAscent", "GeneratorFeatureUnlearning", "GeneratorMachineUnlearning"]): The name of the unlearning algorithm to use.
        dataset_name (Literal["mnist", "cmnist", "fashion_mnist"]): The name of the dataset to use.
        args (dict, optional): Additional arguments to pass to the Unlearner. Defaults to None.

    Returns:
        Dict[str, Unlearner]: A dictionary of Unlearners, keyed by model name.
    """
    
    if name == "SimpleGradientAscent":
        return {k: SimpleGradientAscent(model=v, unlearned_data=args["u_data"], dataset_name=dataset_name) for k, v in args["models"].items()}
    elif name == "GeneratorFeatureUnlearning":
        return {k: GeneratorFeatureUnlearning(model=v, dataset_name=dataset_name) for k, v in args["models"].items()}
    elif name == "GeneratorMachineUnlearning":
        return {k: GeneratorMachineUnlearning(model=v, dataset_name=dataset_name) for k, v in args["models"].items()}
    else:
        raise Exception(f"Unlearning algorithm '{name}' not supported.")

def unlearn_n_models(
        models: Dict[str, torch.nn.Module],
        unlearned_data: DataLoader,
        dataset_name: Literal["mnist", "cmnist", "fashion_mnist"],
        which_unlearning: Literal["SimpleGradientAscent", "GeneratorFeatureUnlearning", "GeneratorMachineUnlearning"],
        data: DataLoader = None,
        logs: bool = True,
        ) -> Dict[str, torch.nn.Module]:
    """
    Unlearns a set of models using the specified unlearning algorithm.

    Parameters:
        models (Dict[str, torch.nn.Module]): A dictionary of models to be unlearned, keyed by model name.
        unlearned_data (DataLoader): DataLoader containing the data to be unlearned from the models.
        dataset_name (Literal["mnist", "cmnist", "fashion_mnist"]): The name of the dataset the models were trained on.
        which_unlearning (Literal["SimpleGradientAscent", "GeneratorFeatureUnlearning", "GeneratorMachineUnlearning"]): The unlearning algorithm to use.
        data (DataLoader, optional): Additional DataLoader for retained data, if needed by the unlearning algorithm.
        logs (bool, optional): Whether to print logs during the unlearning process. Defaults to True.

    Returns:
        Dict[str, torch.nn.Module]: A dictionary of unlearned models, keyed by model name.
    """
    
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
            print(f"Unlearned model {k+1:2}/{len(unlearners):2}...")

    return new_models
