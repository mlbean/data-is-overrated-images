import os
import torch
import random
import numpy as np
import torchvision
from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, Subset, random_split
from torchvision import datasets, transforms

#PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'


class DataParams(NamedTuple):
    total_size: int
    channels: int
    height: int
    width: int
    mean: tuple
    std: tuple
    checkpath: Path

DATASETS = {
    'MNIST': DataParams(total_size=60000, channels=1, height=28, width=28, mean=(0.1307,), std=(0.3081,), checkpath=DATA_DIR / 'MNIST' / 'raw' / 'train-images-idx3-ubyte'),
    'CIFAR10': DataParams(total_size=50000, channels=3, height=32, width=32, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616), checkpath=DATA_DIR / 'cifar-10-batches-py' / 'data_batch_1'),
}

   
def set_all_seeds(seed):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class Data:
    '''
    Simple data module to handle downloading, splitting into train/val/test and building DataLoaders.
    Plots sample images from chosen DataLoader.
    
    Provide custom transforms for training and evaluation by overriding `get_train_transform` 
    and `get_eval_transform`.
    
    Example
    -------
    >>> data = Data('CIFAR10')
    >>> data.setup()
    >>> train_loader = data.train_dataloader()
    >>> val_loader = data.val_dataloader()
    >>> test_loader = data.test_dataloader()
    >>> data.plot_samples(train_loader)
    '''

    def __init__(self, dataset_name, batch_size=128, val_size=5000, seed=42, samplesize=None):
        assert dataset_name in DATASETS, f"Unsupported dataset: {dataset_name}"
        self.params = DATASETS[dataset_name]
        self.dataset_name = dataset_name
        self.dataset_cls = getattr(datasets, dataset_name)
        self.data_dir = DATA_DIR
        self.batch_size = batch_size
        self.val_size = val_size
        self.seed = seed
        self.samplesize = samplesize
        self.g = torch.Generator().manual_seed(seed)
        self.train_idx = None
    
    def get_eval_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize(*NORMS[self.dataset_name]),
        ])

    def get_train_transform(self):
        return transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(*NORMS[self.dataset_name]),
        ])
        
    def _data_exists(self):
        exists = os.path.exists(self.params.checkpath)
        print(f"Checking if data exists at {self.params.checkpath}: {exists}")
        return exists

    def setup(self):
        set_all_seeds(self.seed)

        download = not self._data_exists()
        
        train_transform = self.get_train_transform()
        eval_transform = self.get_eval_transform()
        
        full_train = self.dataset_cls(root=self.data_dir, train=True, download=download, transform=None)
        train_size = len(full_train) - self.val_size
        train_indices, val_indices = random_split(range(len(full_train)), [train_size, self.val_size], generator=self.g)

        if self.samplesize and self.samplesize < train_size:
            print('Training on a subset of size:', self.samplesize)
            self.train_idx = train_indices.indices[:self.samplesize]
        else: 
            self.train_idx = train_indices.indices
            
        self.train_set = Subset(
            self.dataset_cls(root=self.data_dir, train=True, transform=train_transform),
            self.train_idx,
        )
        self.val_set = Subset(
            self.dataset_cls(root=self.data_dir, train=True, transform=eval_transform),
            val_indices.indices,
        )
        self.test_set = self.dataset_cls(root=self.data_dir, train=False, download=download,
                                         transform=eval_transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, generator=self.g)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def plot_samples(self, loader, n=16):
        images, labels = next(iter(loader))
        grid = torchvision.utils.make_grid(images[:64], nrow=8, normalize=True)
    
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class Noise:
    def __init__(self, dataset_name, batch_size=64, val_size=5000, seed=42):
        assert dataset_name in DATASETS, f"Unsupported dataset: {dataset_name}"
        self.dataset_name = dataset_name
        self.params = DATASETS[dataset_name]
        self.data_dir = DATA_DIR
        self.batch_size = batch_size
        self.val_size = val_size
        self.seed = seed

    def setup(self):
        set_all_seeds(self.seed)
        size = (self.params.total_size - self.val_size, self.params.channels, self.params.height, self.params.width)
        
        self.labels = torch.zeros(self.params.total_size - self.val_size, dtype=torch.long)        
        self.noise = torch.rand(size)

    def train_dataloader(self):
        tensorset_tr = TensorDataset(self.noise, self.labels)
        return DataLoader(tensorset_tr, batch_size=self.batch_size, shuffle=False)
  
    def plot_samples(self, loader):
        images, _ = next(iter(loader))
        grid = torchvision.utils.make_grid(images[:64], nrow=8, normalize=True)
    
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
