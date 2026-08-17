
import numpy as cp
#import cupy as cp
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

    def pop(self):
        return self.memory.pop()

# ---------------------------------------------------------
# Weight Initializations
# ---------------------------------------------------------

def xavier_initialization(in_features, out_features):
    k = cp.sqrt(1/(in_features)) 
    rng = cp.random.default_rng() 
    weight = rng.uniform(-k,k, size=(in_features, out_features))
    bias = rng.uniform(-k,k, size=(out_features,))

    return weight, bias

def simple_initialization(in_features, out_features):
    rng = cp.random.default_rng(seed=42)
    weight = rng.normal(size=(in_features, out_features)) * cp.sqrt(1.0 / in_features)
    bias = rng.normal(size=(out_features,)) * cp.sqrt(1.0 / in_features)

    return weight, bias
    


# ---------------------------------------------------------
# Optimizers
# ---------------------------------------------------------

class Adam:
    def __init__(self, trainable_layers, batch_size):

        self.trainable_layers = trainable_layers
        self.num_trainable_layers = len(self.trainable_layers)
        self.batch_size = batch_size

        # Shared for weights and bias and layers
        self.e = 1e-8
        self.lr = 0.001 # default: 0.001
        self.b_1 = 0.9
        self.b_1_t = self.b_1
        self.b_2 = 0.999
        self.b_2_t = self.b_2

        # For initialization of lists
        init_list = cp.zeros(self.num_trainable_layers)

        # For weights
        self.m_w = list(init_list)
        self.v_w = list(init_list)
            
        # For bias
        self.m_b = list(init_list)
        self.v_b = list(init_list)

        # For output
        self.gradients = list(init_list)




    def step(self, grads):
        for i in range(self.num_trainable_layers):

            # Weight update
            self.m_w[i] = self.b_1 * self.m_w[i] +(1-self.b_1) * grads[i][0] / self.batch_size
            self.v_w[i] = self.b_2 * self.v_w[i] + (1-self.b_2) * (grads[i][0] / self.batch_size)**2  
            m_hat_w = self.m_w[i] / (1-self.b_1_t) 
            v_hat_w = self.v_w[i] / (1-self.b_2_t)
            grad_w = m_hat_w / (cp.sqrt(v_hat_w) + self.e)
                   
            
            # Bias update
            self.m_b[i] = self.b_1 * self.m_b[i] +(1-self.b_1) * grads[i][1] / self.batch_size
            self.v_b[i] = self.b_2 * self.v_b[i] + (1-self.b_2) * (grads[i][1] / self.batch_size)**2  
            m_hat_b = self.m_b[i] / (1-self.b_1_t) 
            v_hat_b = self.v_b[i] / (1-self.b_2_t)
            
            grad_b = m_hat_b / (cp.sqrt(v_hat_b) + self.e)


            self.gradients[i] = (grad_w, grad_b)

            
        # Shared parameter update
        self.b_1_t = self.b_1_t*self.b_1
        self.b_2_t = self.b_2_t*self.b_2

            

        return self.lr, self.gradients


class PrimitiveOptimizer:
    # possible augmentation: 1. constant learning rate at first and 2. learning rate reset
    def __init__(self, trainable_layers, batch_size):
        self.lr = 0.1 
        self.steps = 0
        self.learning_rate_history = []

        self.learning_rate_reductions = 50 # after how many epochs we decrease the learning rate
        self.learning_rate_divisor = 1.2 # the number which divides the learning rate, default: 
        

    def step(self, grads):
        self.steps += 1

        
        if(self.steps % (self.learning_rate_reductions) == 0):  
            self.lr = self.lr/self.learning_rate_divisor 

        # For plotting the learning rate
        self.learning_rate_history.append(self.lr)

        return (self.lr, grads)

    def learning_rate_history_get(self):
        return self.learning_rate_history