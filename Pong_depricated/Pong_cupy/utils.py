
import numpy as np
import cupy as cp
from collections import deque
import random

# ---------------------------------------------------------
# Loss and gradient of loss
# ---------------------------------------------------------

def compute_loss_mse(target: cp.ndarray, prediction: cp.ndarray) -> float:
    """Return MSE"""
    return cp.sum((target -prediction)**2) / prediction.shape[0] 


def compute_gradient(target: cp.ndarray, prediction: cp.ndarray) -> cp.ndarray:
    """
    Computes the gradient of the mse error.
    """
    return 2*(prediction - target) / prediction.shape[0]
# ---------------------------------------------------------    

# ---------------------------------------------------------
# Replay Memory
# ---------------------------------------------------------
# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# ---------------------------------------------------------
# Weight Initializations
# ---------------------------------------------------------

def xavier_initialization(in_features, out_features):
    k = cp.sqrt(1/(in_features)) 
    rng = cp.random.default_rng() 
    weight = rng.uniform(-k,k, size=(in_features, out_features))
    bias = rng.uniform(-k,k, size=(out_features,))

    return weight, bias

def other_initialization(in_features, out_features):
    rng = cp.random.default_rng(seed=42)
    weight = rng.normal(size=(in_features, out_features)) * cp.sqrt(1.0 / in_features)
    bias = rng.normal(size=(out_features,)) * cp.sqrt(1.0 / in_features)

    return weight, bias
    


# ---------------------------------------------------------
# Optimizers
# ---------------------------------------------------------
#TODO PRIMITIVE + ADAM

class Adam:
    def __init__(self, lr = 0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8

    def update_parameters(self, layer):
        pass

