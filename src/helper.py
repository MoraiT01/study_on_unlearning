"""Helper function for the project."""

from typing import Literal, Tuple, Dict
from torch.utils.data import DataLoader

from mlp_dataclass import MNIST_CostumDataset

def get_dataset_subsetloaders(dataset_name: Literal["mnist", "cmnist", "fashion_mnist"] = "mnist") -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, DataLoader]]:
    """
    This function creates a tuple of four DataLoaders which can be used to load the
    dataset in different ways.

    Returns:
        D_GESAMT: A DataLoader which loads the whole dataset
        D_ERASED: A DataLoader which loads the erased samples
        D_REMAIN: A DataLoader which loads the remaining samples
        D_CLASSES: A dictionary which contains DataLoaders for the different classes
    """
    # Making sure it is downloaded
    MNIST_CostumDataset(dataset_name=dataset_name, download=True)

    D_GESAMT = DataLoader(
        dataset=MNIST_CostumDataset(
            sample_mode="all",
            train=True,
            test=True,
            dataset_name=dataset_name,
        ),
        batch_size=1,
        shuffle=False,
    )

    D_ERASED = DataLoader(
        dataset=MNIST_CostumDataset(
            sample_mode="only_erased",
            train=True,
            test=True,
            dataset_name=dataset_name,
        ),
        batch_size=1,
        shuffle=False,
    )

    D_REMAIN = DataLoader(
        dataset=MNIST_CostumDataset(
            sample_mode="except_erased",
            train=True,
            test=True,
            dataset_name=dataset_name,
        ),
        batch_size=1,
        shuffle=False,
    )

    D_CLASSES = {k: DataLoader(dataset=MNIST_CostumDataset(sample_mode="all", classes=[k], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False) for k in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]}

    # Let's differantiate between the Erased and the Remain of the respective class too
    if dataset_name == "mnist" or dataset_name == "cmnist":
        D_CLASSES["7_all"]      = D_CLASSES.pop("7")
        D_CLASSES["7_remain"]   = DataLoader(dataset=MNIST_CostumDataset(sample_mode="except_erased", classes=["7"], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False)
        D_CLASSES["7_erased"]   = DataLoader(dataset=MNIST_CostumDataset(sample_mode="only_erased"  , classes=["7"], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False)
    elif dataset_name == "fashion_mnist":
        D_CLASSES["5_all"]      = D_CLASSES.pop("5")
        D_CLASSES["5_remain"]   = DataLoader(dataset=MNIST_CostumDataset(sample_mode="except_erased", classes=["5"], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False)
        D_CLASSES["5_erased"]   = DataLoader(dataset=MNIST_CostumDataset(sample_mode="only_erased"  , classes=["5"], train=True, test=True, dataset_name=dataset_name), batch_size=1, shuffle=False)
    else:
        raise Exception(f"Dataset '{dataset_name}' not supported.")

    return D_GESAMT, D_ERASED, D_REMAIN, D_CLASSES