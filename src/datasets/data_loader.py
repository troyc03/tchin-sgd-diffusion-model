# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:18:33 2026

@author: Troy
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torch.nn import functional as F
from torchvision.datasets import MNIST, CIFAR10


def load_mnist(batch_size=128, train=True):
    """Load the MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

def load_cifar10(batch_size=128, train=True):
    """Load the CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)


def get_dataset(name: str):
    """Return a dataset object by name."""
    if name.lower() == 'mnist':
        return load_mnist()
    elif name.lower() == 'cifar10':
        return load_cifar10()
    else:
        raise NotImplementedError("Dataset not implemented: {}".format(name))
