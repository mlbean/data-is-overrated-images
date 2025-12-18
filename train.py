from dataset import Data
from model import CAE

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm 
import matplotlib.pyplot as plt


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')


def train(settings, hyperparams):
    dataset = settings['dataset']
    max_epochs = settings['max_epochs']
    lr = hyperparams['lr']
    criterion = nn.MSELoss()
    data = Data(dataset)
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    
    train_losses = []
    val_losses = []
    
    model = CAE()
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    for epoch in range(max_epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        total_train_loss = 0
        for x, _ in train_bar:
            x = x.to(DEVICE)
            
            x_hat = model(x)
            loss = criterion(x_hat, x)    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(DEVICE)
                x_hat = model(x)
                loss = criterion(x_hat, x)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)


    return model, train_losses, val_losses


settings = {
    'dataset': 'MNIST',
    'max_epochs': 3,  
}

hyperparams = {
    'lr': 0.001
}

model, train_losses, val_losses = train(settings, hyperparams)

plt.plot(range(1, len(train_losses)+1), train_losses)
plt.plot(range(1, len(val_losses)+1), val_losses)
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.show()
print(1)