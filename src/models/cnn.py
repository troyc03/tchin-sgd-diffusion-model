import torch
from torch.nn import functional as F
import torch.nn as nn
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt


class SimpleCNN(nn.Module):
    def __init__(self):
        pass