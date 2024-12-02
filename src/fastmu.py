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
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.start_dims = 100

        self.f1 = nn.Linear(self.start_dims, 1000)
        self.f2 = nn.Linear(1000, math.prod(self.dim))

    # forward pass   
    def forward(self):
        # random starting noise
        x = torch.randn(self.start_dims)
        x = x.flatten()

        # from noise to learnable patterns
        x = self.f1(x)
        x = torch.relu(x)
        x = self.f2(x)

        # Shape back
        reshaped_tensor = x.view(self.dim)
        return reshaped_tensor

def train_noise_generator(forget_data: Dataset, model: torch.nn.Module) -> Tuple[NoiseGen, Dict, Dict]:

    noises = {}
    og_labels = {}
    created_labels =  {}
    for index, data in enumerate(DataLoader(forget_data, batch_size=1, shuffle=False)): # iterate over forget_data):
        s, l = data

        new_l = F.softmax(model(s.to(DEVICE)), dim=1)
        
        created_labels[index] = new_l[0].to(DEVICE)
        og_labels[index] = new_l.to(DEVICE)
        # Ursprünglich waren hier die Labels der der Klassen gemeint
        # Jedoch entschied ich mich dagegen
        # Der prognostizierte Wahrkeitsvektor ist eine andere Darstellung des Samples,
        # Wir wollen nicht die Klasse unlearnen, sonder das Sample/das Feature

    noises = NoiseGen(s[0].shape).to(DEVICE)

    # the second Dataloader contains the new labels
    # Shuffle True, to create more variance
    c = created_labels

    # original labels shall be kept too
    o = og_labels

    return noises, c, o

def noise_maximization(forget_data: Dataset, model: torch.nn.Module, logs: bool = False) -> Tuple[NoiseGen, torch.Tensor]:
    
    noise_generator, created_labels, original_labels = train_noise_generator(forget_data, model)
    created_l_loader = DataLoader(
        created_labels,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model.to(DEVICE)
    optimizers = torch.optim.Adam(noise_generator.parameters(), lr = 0.1) # Hyperparameter
    # Optional learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizers, step_size=5, gamma=0.1)

    num_epochs = 5 # Hyperparameter
    step_size  = 100 # Hyperparameter

    # TODO überarbeiten, NoiseGenerator sollte ...
    for epoch in range(num_epochs):
        total_loss = []
        step = 0
        for l in created_l_loader: # Der Loop ist unnötig
            # Firstly, generate the noise
            input_batch = None
            for i in range(BATCH_SIZE):	# My Standard Batchsize
                new = noise_generator().to(DEVICE)
                input_batch = new if input_batch == None else torch.cat((input_batch, new), dim=0)

            outputs = model(input_batch)
            loss = - F.cross_entropy(outputs, l) + 0.1 * torch.mean(torch.sum(torch.square(input_batch)))
            loss.backward(retain_graph=True) # Sollte weg, da die Schleife weg soll

            # for logging
            total_loss.append(loss.cpu().detach().numpy())

            step += 1
            if step >= step_size:
                step = 0
                scheduler.step()
            
        print("Epoch: {}, Loss: {}".format(epoch, np.mean(total_loss)))

    created_labels = torch.stack(list(created_labels.values()))
    created_label = torch.mean(created_labels, dim=0)
    
    return noise_generator, created_label

class FeatureMU_Loader(Dataset):
    def __init__(self, noise_generator,label4noise, number_of_noise, retain_data):
        self.noise_gen = noise_generator
        self.retain_data = retain_data
        self.number_of_noise = number_of_noise
        self.label4noise = label4noise

    def __len__(self):
        return len(self.number_of_noise) + len(self.retain_data)

    def __getitem__(self, idx):
        if idx < self.number_of_noise:
            return self.noise_gen(), self.label4noise
        else:
            return self.retain_data.__getitem__(idx - self.n)

def impairing_phase(noise_generator: NoiseGen, number_of_noise: int, label4noise: torch.Tensor, retain_loader: Dataset, model: torch.nn.Module, logs: bool = False) -> torch.nn.Module:

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

    model.to(DEVICE)

    # Validation Dataloaders
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
    data_forget = MNIST_CostumDataset(
        sample_mode="only_erased",
        train=True,
        test=False,
        dataset_name=dataset_name,
    )

    # (Re)Training Dataloaders
    noise_gen, noise_create_label = noise_maximization(
        forget_data=data_forget,
        model=model,
        logs=logs,
    )

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
        label4noise=noise_create_label,
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