import numpy as np
import torch

class ParameterDiffusionAnalyzer:
    def __init__(self, model, data_loader, loss_fn):
       self.model = model
       self.model_parameters = list(model.parameters())
       self.data_loader = data_loader
       self.loss_fn = loss_fn
       pass