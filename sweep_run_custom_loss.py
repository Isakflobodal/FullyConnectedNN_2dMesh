import wandb
from NN import NeuralNet
import torch
from dataset import ContourDataset
from trainer import train
from torch.utils.data import DataLoader
from torch import autograd, dropout
import numpy as np
import random 
import wandb


# Seed
seed = 11
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
from createdata import dim

id = 'cc4fhozv'


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run():
    with wandb.init(project="2d_fully_connected", entity="master-thesis-ntnu", config=dict) as run:
        config = wandb.config
        
        df_loss_value = config.df_loss_value
        layer_sizes = [512,1024,1024,512, dim**2] 
        num_epochs = 1
        batch_size = 16  
        learning_rate = 0.008
        dropout = 0.0
    

        use_existing_model = True
        # Dataset
        train_dataset = ContourDataset()
        test_dataset = ContourDataset(split="test")
        validation_dataset = ContourDataset(split="validation")
        # Data loader
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True,  pin_memory=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
        validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True, pin_memory=True)
        
        model = NeuralNet(layer_sizes, dropout).to(device)
        if use_existing_model:
            model.load_state_dict(torch.load("model_onlyL1.pth")["state_dict"])
        #with autograd.detect_anomaly():
        wandb.init(project="2d_fully_connected", entity="master-thesis-ntnu", config=dict)

        train(model, num_epochs, batch_size, train_loader, test_loader, validation_loader, df_loss_value, learning_rate=learning_rate, device=device)
        
        # Save model
        config = {
            "state_dict": model.state_dict()
        }

        torch.save(config, "model.pth") 



if __name__ == "__main__":
    wandb.agent(id, project="2d_fully_connected", entity="master-thesis-ntnu", function=run, count=7)