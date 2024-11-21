"""contains all the code necessary for running the MU algorithm proposed by https://github.com/vikram2000b/Fast-Machine-Unlearning"""

# import required libraries
import numpy as np
import tarfile
import os

from typing import Literal, Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tt
from torchvision.models import resnet18

# my
from training import model_params

torch.manual_seed(100)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def training_step(model, batch):
    images, labels = batch
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    out = model(images)                  
    loss = F.cross_entropy(out, labels) 
    return loss

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

def epoch_end(model, epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['Loss'], result['Acc']))
    
def distance(model,model0):
    distance=0
    normalization=0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        space='  ' if 'bias' in k else ''
        current_dist=(p.data0-p0.data0).pow(2).sum().item()
        current_norm=p.data0.pow(2).sum().item()
        distance+=current_dist
        normalization+=current_norm
    print(f'Distance: {np.sqrt(distance)}')
    print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    return 1.0*np.sqrt(distance/normalization)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, model, train_loader, val_loader, optimizer):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = optimizer

    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
 
    for epoch in range(epochs): 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch)
            train_losses.append(loss)
            loss.backward()
        
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
                    
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        sched.step(result['Loss'])
    return history


# defining the noise structure
class Noise(nn.Module):
    def __init__(self, samples):
        super().__init__()
        self.noise = torch.nn.Parameter(samples, requires_grad = True)
        
    def forward(self):
        return self.noise

class ToForget(Dataset):
    def __init__(
            self,
            forget_data: DataLoader,
            model: torch.nn.Module,
            dataset_name: str,
            ):
        super().__init__()

        self.lr = model_params[dataset_name]["lr"]
        self.samples = self._initialize(forget_data, model)

    def _initialize(self, forget_data, model) -> Dataset:

        new_s = {}
        for k, v in self.samples.items():
            new_s[len(new_s)] = self.noise_maxing_per_batch(batch=v, model=model, n_epochs=5)


        return new_s

    def noise_maxing_per_batch(self, batch: Tuple[torch.Tensor, torch.Tensor], model: torch.nn.Module, n_epochs: int = 5, logs: False) -> torch.Tensor:

        # "Optiming loss for class {}".format(cls))
        noises = Noise(batch[0]).to(DEVICE)
        labels = batch[1]

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(noises.parameters(), lr=self.lr, maximize=True)

        num_steps = 8
        class_label = cls
        for epoch in range(n_epochs):
            total_loss = []
            
            outputs = model(inputs)
            loss = - loss_function(outputs, labels) + 0.1 * torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss.append(loss.cpu().detach().numpy())

            if logs:
                print("Loss: {}".format(np.mean(total_loss)))

        return None
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
    
# list of all classes
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# classes which are required to un-learn
classes_to_forget = [0, 2]

# classwise list of samples
num_classes = 10
classwise_train = {}
for i in range(num_classes):
    classwise_train[i] = []

for img, label in train_ds:
    classwise_train[label].append((img, label))
    
classwise_test = {}
for i in range(num_classes):
    classwise_test[i] = []

for img, label in valid_ds:
    classwise_test[label].append((img, label))

# getting some samples from retain classes
num_samples_per_class = 1000

retain_samples = []
for i in range(len(classes)):
    if classes[i] not in classes_to_forget:
        retain_samples += classwise_train[i][:num_samples_per_class]

# retain validation set
retain_valid = []
for cls in range(num_classes):
    if cls not in classes_to_forget:
        for img, label in classwise_test[cls]:
            retain_valid.append((img, label))
            
# forget validation set
forget_valid = []
for cls in range(num_classes):
    if cls in classes_to_forget:
        for img, label in classwise_test[cls]:
            forget_valid.append((img, label))
            
forget_valid_dl = DataLoader(forget_valid, batch_size, num_workers=3, pin_memory=True)
retain_valid_dl = DataLoader(retain_valid, batch_size*2, num_workers=3, pin_memory=True)

# loading the model
model = resnet18(num_classes = 10).to(device = device)
model.load_state_dict(torch.load("ResNET18_CIFAR10_ALL_CLASSES.pt"))


### Noise Maximization Generation
%%time

noises = {}
for cls in classes_to_forget:
    print("Optiming loss for class {}".format(cls))
    noises[cls] = Noise(batch_size, 3, 32, 32).cuda()
    opt = torch.optim.Adam(noises[cls].parameters(), lr = 0.1)

    num_epochs = 5
    num_steps = 8
    class_label = cls
    for epoch in range(num_epochs):
        total_loss = []
        for batch in range(num_steps):
            inputs = noises[cls]()
            labels = torch.zeros(batch_size).cuda()+class_label
            outputs = model(inputs)
            loss = -F.cross_entropy(outputs, labels.long()) + 0.1*torch.mean(torch.sum(torch.square(inputs), [1, 2, 3]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss.append(loss.cpu().detach().numpy())
        print("Loss: {}".format(np.mean(total_loss)))


### Impair
%%time

batch_size = 256
noisy_data = []
num_batches = 20
class_num = 0

for cls in classes_to_forget:
    for i in range(num_batches):
        batch = noises[cls]().cpu().detach()
        for i in range(batch[0].size(0)):
            noisy_data.append((batch[i], torch.tensor(class_num)))

other_samples = []
for i in range(len(retain_samples)):
    other_samples.append((retain_samples[i][0].cpu(), torch.tensor(retain_samples[i][1])))
noisy_data += other_samples
noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=256, shuffle = True)


optimizer = torch.optim.Adam(model.parameters(), lr = 0.02)


for epoch in range(1):  
    model.train(True)
    running_loss = 0.0
    running_acc = 0
    for i, data in enumerate(noisy_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(),torch.tensor(labels).cuda()

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
    print(f"Train loss {epoch+1}: {running_loss/len(train_ds)},Train Acc:{running_acc*100/len(train_ds)}%")


### Repair
%%time

heal_loader = torch.utils.data.DataLoader(other_samples, batch_size=256, shuffle = True)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


for epoch in range(1):  
    model.train(True)
    running_loss = 0.0
    running_acc = 0
    for i, data in enumerate(heal_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(),torch.tensor(labels).cuda()

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
    print(f"Train loss {epoch+1}: {running_loss/len(train_ds)},Train Acc:{running_acc*100/len(train_ds)}%")

def noise_maximization() -> DataLoader:
    pass

def impairing_phase() -> torch.nn.Module:
    pass

def repairing_phase() -> torch.nn.Module:
    pass

def _main(
        model: torch.nn.Module,
        forget_data: DataLoader,
        retain_data: DataLoader,
        dataset_name: Literal["mnist", "cmnist", "fashion_mnist"],
    ) -> torch.nn.Module:
    model.to(DEVICE)

    noise_maximization()

    impairing_phase()

    print("Performance of Standard Forget Model on Forget Class")
    history = [evaluate(model, forget_valid_dl)]
    print("Accuracy: {}".format(history[0]["Acc"]*100))
    print("Loss: {}".format(history[0]["Loss"]))

    print("Performance of Standard Forget Model on Retain Class")
    history = [evaluate(model, retain_valid_dl)]
    print("Accuracy: {}".format(history[0]["Acc"]*100))
    print("Loss: {}".format(history[0]["Loss"]))


    repairing_phase()

    print("Performance of Standard Forget Model on Forget Class")
    history = [evaluate(model, forget_valid_dl)]
    print("Accuracy: {}".format(history[0]["Acc"]*100))
    print("Loss: {}".format(history[0]["Loss"]))

    print("Performance of Standard Forget Model on Retain Class")
    history = [evaluate(model, retain_valid_dl)]
    print("Accuracy: {}".format(history[0]["Acc"]*100))
    print("Loss: {}".format(history[0]["Loss"]))

    new_model = None
    return new_model