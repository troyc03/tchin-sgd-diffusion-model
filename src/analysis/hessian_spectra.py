import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class Hessian:
    def __init__(self, model, loss_fn, data_loader):
        self.model = model 
        self.loss_fn = loss_fn
        self.data_loader = data_loader
    
    def compute_hessian(self):
        pass