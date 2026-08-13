
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