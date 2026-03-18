import numpy as np
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
import torch

class GradientNoiseAnalyzer:
    def __init__(self, model, data_loader, loss_fn):
        self.model = model
        self.data_loader = data_loader
        self.loss_fn = loss_fn

    pass