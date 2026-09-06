
import numpy as np
from collections import deque
import random

# ---------------------------------------------------------
# Loss and gradient of loss
# ---------------------------------------------------------

def compute_loss_mse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Calculation of MSE loss"""
    return np.sum((target -prediction)**2) / prediction.shape[0] 


def compute_gradient(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """ Calculation of Gradient of the mse error """
    return 2*(prediction - target) / prediction.shape[0]


# ---------------------------------------------------------
# Replay Memory
# ---------------------------------------------------------

class ReplayMemory():
    """The replay memory"""
    def __init__(self, maxlen):
        # Initialization
        self.memory = deque([], maxlen=maxlen)
        
    def append(self, transition):
        # Adding new transition to replay memory
        self.memory.append(transition)

    def sample(self, sample_size):
        # Sampling of mini batch from replay memory
        return random.sample(self.memory, sample_size)

    def __len__(self):
        # Returns the number of transitions saved
        return len(self.memory)

    def pop(self):
        # Returns and removes newest state transition
        return self.memory.pop()

# ---------------------------------------------------------
# Weight Initializations
# ---------------------------------------------------------
def xavier_initialization(in_features, out_features, in_states, out_actions):
    """One version of the Xavier weight initialization"""

    # Initializing the distribution 
    k = np.sqrt(1/(in_states)) 
    rng = np.random.default_rng(seed = 42) 

    # Sampling weight and bias values
    weight = rng.uniform(-k,k, size=(in_features, out_features))
    bias = rng.uniform(-k,k, size=(out_features,))

    return weight, bias

def old_xavier_initialization(in_features, out_features, in_states, out_actions):
    """One version of the Xavier weight initialization which we initially used"""

    # Initializing the distribution 
    k = np.sqrt(1/(in_features)) 
    rng = np.random.default_rng(seed = 42) 

    # Sampling weight and bias values
    weight = rng.uniform(-k,k, size=(in_features, out_features))
    bias = rng.uniform(-k,k, size=(out_features,))

    return weight, bias

def xavier_initialization_uniform(in_features, out_features, in_states, out_actions):
    """A version of the Xavier weight initialization using uniform distribution"""

    # Initializing the distribution 
    k = np.sqrt(6/(in_states+out_actions)) 
    rng = np.random.default_rng(seed = 42) 

    # Sampling weight and bias values
    weight = rng.uniform(-k,k, size=(in_features, out_features))
    bias = rng.uniform(-k,k, size=(out_features,))

    return weight, bias

def xavier_initialization_normal(in_features, out_features, in_states, out_actions):
    """A version of the Xavier weight initialization using normal distribution"""

    # Initializing the distribution 
    k = np.sqrt(2/(in_states+out_actions)) 
    rng = np.random.default_rng(seed = 42) 

    # Sampling weight and bias values
    weight = rng.normal(0,k, size=(in_features, out_features))
    bias = rng.normal(0,k, size=(out_features,))

    return weight, bias

def simple_initialization(in_features, out_features, in_states, out_actions):
    """The weight initialization provided in the lecture"""

    # Initializing the distribution 
    rng = np.random.default_rng(seed = 42)
    k = np.sqrt(1.0 / in_features)

    # Sampling weight and bias values
    weight = rng.normal(size=(in_features, out_features)) * k
    bias = rng.normal(size=(out_features,)) * k

    return weight, bias

def kaiming_initialization(in_features, out_features, in_states, out_actions):
    """An implementation of the Kaiming weight initialization"""

    # Initializing the distribution 
    k = np.sqrt(2/(in_features)) 
    rng = np.random.default_rng(seed = 42) 

    # Sampling weight and bias values
    weight = rng.normal(0,k, size=(in_features, out_features))
    bias = rng.normal(0,k, size=(out_features,))

    return weight, bias
    


# ---------------------------------------------------------
# Optimizers
# ---------------------------------------------------------

class Adam:
    """The Adam optimizer"""
    def __init__(self, trainable_layers, batch_size):
        # Initialization
        self.name = "Adam"
        self.trainable_layers = trainable_layers
        self.num_trainable_layers = len(self.trainable_layers)
        self.batch_size = batch_size

        # Parameters shared for weights and bias and layers
        self.lr = 0.001 # default: 0.001
        self.b_1 = 0.9 # default: 0.9
        self.b_2 = 0.999 # default: 0.999

        self.e = 1e-8

        # Variables for calculation of bias correction
        self.b_1_t = self.b_1
        self.b_2_t = self.b_2


        # Initialization of lists of gradient averages of every layer

        # Length of lists
        init_list = np.zeros(self.num_trainable_layers)

        # Lists for averages of weights
        self.m_w = list(init_list)
        self.v_w = list(init_list)
            
        # Lists for averages of bias
        self.m_b = list(init_list)
        self.v_b = list(init_list)

        # List of corrected gradients
        self.gradients = list(init_list)




    def step(self, grads):
        # Calculation of weight and bias update value for every trainable layer
        for i in range(self.num_trainable_layers):

            # Weight update calculation
            self.m_w[i] = self.b_1 * self.m_w[i] +(1-self.b_1) * grads[i][0] / self.batch_size
            self.v_w[i] = self.b_2 * self.v_w[i] + (1-self.b_2) * (grads[i][0] / self.batch_size)**2  
            m_hat_w = self.m_w[i] / (1-self.b_1_t) 
            v_hat_w = self.v_w[i] / (1-self.b_2_t)
            grad_w = m_hat_w / (np.sqrt(v_hat_w) + self.e)
                   
            
            # Bias update calculation
            self.m_b[i] = self.b_1 * self.m_b[i] +(1-self.b_1) * grads[i][1] / self.batch_size
            self.v_b[i] = self.b_2 * self.v_b[i] + (1-self.b_2) * (grads[i][1] / self.batch_size)**2  
            m_hat_b = self.m_b[i] / (1-self.b_1_t) 
            v_hat_b = self.v_b[i] / (1-self.b_2_t)
            grad_b = m_hat_b / (np.sqrt(v_hat_b) + self.e)


            # Saves the calculated weight and bias update value in list
            self.gradients[i] = (grad_w, grad_b)

            
        # Shared parameter update
        self.b_1_t = self.b_1_t*self.b_1
        self.b_2_t = self.b_2_t*self.b_2

        # Returns learning rate and calculated weight and bias update value
        return self.lr, self.gradients
    
    def get_name(self):
        # Returns the name of the optimizer
        return self.name


class PrimitiveOptimizer:
    """Implementation of a primitive optimizer"""
    def __init__(self, trainable_layers, batch_size):
        # Initialization
        self.name = "Primitive Optimizer"
        self.steps = 0

        # For plotting the learning rate if needed
        self.learning_rate_history = []

        # Hyperparameters
        self.lr = 0.1 # learning rate
        self.learning_rate_reductions = 50 # number epochs  after which learning rate decreases
        self.learning_rate_divisor = 1.2 # the number which divides learning rate 
        

    def step(self, grads):
        # Lowers the learning rate after a certain amount of epochs passed
        self.steps += 1
        if(self.steps % (self.learning_rate_reductions) == 0):  
            self.lr = self.lr/self.learning_rate_divisor 

        # For plotting the learning rate
        self.learning_rate_history.append(self.lr)

        return (self.lr, grads)

    def learning_rate_history_get(self):
        # Returns the learning rate history for plotting
        return self.learning_rate_history
    
    def get_name(self):
        # Returns its name
        return self.name