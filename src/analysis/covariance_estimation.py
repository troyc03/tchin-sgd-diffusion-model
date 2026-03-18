import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch

class CovarianceEstimator:
    
    def __init__(self, data, shrinkage=0.1):
        self.data = data
        self.shrinkage = shrinkage
    
    pass