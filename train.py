from dataset import Data
from model import CAE

import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm 

DEVICE = 'mps'

data = Data('MNIST')
data.setup()

train_loader = data.train_dataloader()

foo = next(iter(train_loader))


def train(settings, hyperparams):
    dataset = settings['dataset']
    max_epochs = settings['max_epochs']
    lr = hyperparams['lr']
    
    data = Data(dataset)
    data.setup()
    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    
    
    model = CAE()
    model = model.to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    for epoch in range(max_epochs):
        model.train()
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        total_train_loss = 0
        for x, _ in train_bar:
            x.to(DEVICE)
            
            x_hat = model(x)
            loss = nn.MSELoss(x_hat, x)    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            train_bar.set_postfix(loss=loss.item())
    return model


settings = {
    'dataset': 'MNIST',
    'max_epochs': 2,  
}

hyperparams = {
    'lr': 0.001
}

model = train(settings, hyperparams)
print(model)