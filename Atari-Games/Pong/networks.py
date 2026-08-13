import numpy as np
import cupy as cp
from layers import Linear, LinearAdam, Sigmoid, Relu, Conv

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        # Preparing input
        x = cp.array(x)
        x = x.reshape(x.shape[0], -1)  # Flatten the input
        x = x.T
        
        for layer in self.layers:
            x = layer.forward(x)
        
        #print(x)
        return x.T #[0] #TODO CHECK WHICH ONE IS CORRECT
    
    def backward(self, x: cp.ndarray) -> None:
        for layer in reversed(self.layers):
            x = layer.backward(x)
    
    def update(self,lr) -> None:
        for layer in self.layers:
            layer.update(lr)
    
    def get_weights(self):
        # Returns weights of the neurons
        return [layer.get_weights() for layer in self.layers]
        
    def set_weights(self, weights):
        # Sets weights of the neurons
        for i in range(len(self.layers)):
            self.layers[i].set_weights(weights[i])


def create_three_layers_model(in_states, h1_nodes, out_actions, batch_size: int, adam: bool):
    if (adam == True):
        model = NeuralNetwork([
            LinearAdam(in_states, h1_nodes, batch_size),   # Linear layer with adam optimizer
            Relu(h1_nodes, batch_size), # Relu activation layer
            LinearAdam(h1_nodes, h1_nodes, batch_size),   # Linear layer with adam optimizer
            Relu(h1_nodes, batch_size), # Relu activation layer
            LinearAdam(h1_nodes, out_actions, batch_size), # Linear layer with adam optimizer
        ])
    else:
        model = NeuralNetwork([
            Linear(in_states, h1_nodes, batch_size),   # Linear layer 
            Relu(h1_nodes, batch_size), # Relu activation layer
            Linear(h1_nodes, h1_nodes, batch_size),   # Linear layer 
            Relu(h1_nodes, batch_size), # Relu activation layer
            Linear(h1_nodes, out_actions, batch_size), # Linear layer 
        ])

    return model


def create_two_layers_model(in_states, h1_nodes, out_actions, batch_size: int, adam: bool):
    if (adam == True):
        model = NeuralNetwork([
            LinearAdam(in_states, h1_nodes, batch_size),   # Linear layer with adam optimizer
            Relu(batch_size), # Relu activation layer
            LinearAdam(h1_nodes, out_actions, batch_size), # Linear layer with adam optimizer
        ])
    else:
        model = NeuralNetwork([
            Linear(in_states, h1_nodes, batch_size),   # Linear layer 
            Relu(batch_size), # Relu activation layer
            Linear(h1_nodes, out_actions, batch_size), # Linear layer 
        ])

    return model


def create_custom_model(in_states, h1_nodes, out_actions, batch_size: int, adam: bool):
    # TODO Test and add toggle adam
    model = NeuralNetwork([
        Conv(n_filters=8, filter_size=(8, 8), padding=0, stride=4),
        Relu(batch_size), # Relu activation layer
        Conv(n_filters=2, filter_size=(4, 4), padding=0, stride=2),
        Relu(batch_size), # Relu activation layer
        Conv(n_filters=1, filter_size=(3, 3), padding=0, stride=1),
        Relu(batch_size), # Relu activation layer
        LinearAdam(3136, 512, batch_size),   # Linear layer with adam optimizer
        Relu(batch_size), # Relu activation layer
        LinearAdam(512, out_actions, batch_size), # Linear layer with adam optimizer
    ])

    
    return model
    
        

    
        

    




   