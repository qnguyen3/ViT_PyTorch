import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR100
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import os
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import ViT

seed = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs: int = 100):
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for data, label in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            #Load data into cuda
            data = data.to(device)
            label = label.to(device)
            #Pass data to model
            output = model(data)
            loss = criterion(output, label)
            #Optimizing
            loss.backward()
            optimizer.step()
            #Calculate Accuracy
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
            if val_loader is not None:
                epoch_val_accuracy = 0
                epoch_val_loss = 0
                for data, label in valid_loader:
                    model.eval()
                    #Load val_data into cuda
                    data = data.to(device)
                    label = label.to(device)
                    #Pass val_data to model
                    val_output = model(data)
                    val_loss = criterion(val_output, label)
                    #Calculate Validation Accuracy
                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_accuracy += acc / len(valid_loader)
                    epoch_val_loss += val_loss / len(valid_loader)
        if val_loader is not None:
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
                )
        else:
            print(
                f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f}\n"
                )





def read_config(config_path: str = "config.txt"):
    with open(config_path) as f:
        lines = f.readlines()
    lines = [word.strip('\n') for word in lines]
    return {'batch_size': int(lines[0]),
            'epochs': int(lines[1]),
            'learning_rate': float(lines[2]),
            'gamma': float(lines[3]),
            'img_size': int(lines[4]),
            'patch_size': int(lines[5]),
            'num_class': int(lines[6]),
            'd_model': int(lines[7]),
            'n_head': int(lines[8]),
            'n_layers': int(lines[9]),
            'd_mlp': int(lines[10]),
            'channels': int(lines[11]),
            'dropout': float(lines[12]),
            'pool': lines[13]}

    


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    seed_everything(seed)
    configs = read_config()
    img_size = configs['img_size']

    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_data = CIFAR100(download=True,root="./cifar100",transform=train_transforms)
    test_val_data = CIFAR100(root="./cifar100",train = False,transform=test_transforms)
    train_len = len(train_data)
    val_len = test_len = int(len(test_val_data) / 2)
    test_data, val_data = torch.utils.data.random_split(test_val_data, [test_len, val_len])
    num_class = len(np.unique(train_data.targets))
    train_loader = DataLoader(dataset = train_data, batch_size = configs['batch_size'], shuffle = True)
    test_loader = DataLoader(dataset = test_data, batch_size=configs['batch_size'], shuffle = True)
    valid_loader = DataLoader(dataset = val_data, batch_size=configs['batch_size'], shuffle = True)
    
    vision_transformer = ViT(img_size = configs['img_size'],
                            patch_size = configs['patch_size'],
                            num_class = configs['num_class'],
                            d_model = configs['d_model'],
                            n_head = configs['n_head'],
                            n_layers = configs['n_layers'],
                            d_mlp = configs['d_mlp'],
                            channels = configs['channels'],
                            dropout = configs['dropout'],
                            pool = configs['pool']).to(device)
    #epochs
    epochs = configs['epochs']
    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(vision_transformer.parameters(), lr=configs['learning_rate'])
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=configs['gamma'])
    
    train(vision_transformer, train_loader, valid_loader, criterion, optimizer, scheduler)
