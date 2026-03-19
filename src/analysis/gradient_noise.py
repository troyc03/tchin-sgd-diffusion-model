import os
import sys

# THIS MUST BE THE FIRST LINE EXECUTED
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch.func import vmap, grad, functional_call
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F

class GradientNoiseAnalyzer:
    pass