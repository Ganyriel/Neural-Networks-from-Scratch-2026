import numpy as np
import cupy as cp
from layers import Linear, LinearAdam, Sigmoid, Relu, Conv, Flatten

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x: cp.ndarray) -> cp.ndarray:
        # Preparing input
        x = cp.array(x)
        
        for layer in self.layers:
            x = layer.forward(x)
        
        return x
    
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

    def print_name(self):
        # Prints the name of every layer
        print("")
        print("Network architecture: ")
        for layer in self.layers:
            layer.print_name()
        print("")
        


def create_three_layers_model(in_states, h1_nodes, out_actions, batch_size: int, adam: bool):
    if (adam == True):
        model = NeuralNetwork([
            Flatten(),
            LinearAdam(in_states, h1_nodes, batch_size),   # Linear layer with adam optimizer
            Relu(), # Relu activation layer
            LinearAdam(h1_nodes, h1_nodes, batch_size),   # Linear layer with adam optimizer
            Relu(), # Relu activation layer
            LinearAdam(h1_nodes, out_actions, batch_size), # Linear layer with adam optimizer
        ])
    else:
        model = NeuralNetwork([
            Flatten(),
            Linear(in_states, h1_nodes, batch_size),   # Linear layer 
            Relu(), # Relu activation layer
            Linear(h1_nodes, h1_nodes, batch_size),   # Linear layer 
            Relu(), # Relu activation layer
            Linear(h1_nodes, out_actions, batch_size), # Linear layer 
        ])

    return model


def create_two_layers_model(in_states, h1_nodes, out_actions, batch_size: int, adam: bool):
    if (adam == True):
        model = NeuralNetwork([
            Flatten(),
            LinearAdam(in_states, h1_nodes, batch_size),   # Linear layer with adam optimizer
            Relu(), # Relu activation layer
            LinearAdam(h1_nodes, out_actions, batch_size), # Linear layer with adam optimizer
        ])
    else:
        model = NeuralNetwork([
            Flatten(),
            Linear(in_states, h1_nodes, batch_size),   # Linear layer 
            Relu(), # Relu activation layer
            Linear(h1_nodes, out_actions, batch_size), # Linear layer 
        ])

    return model


def create_custom_model(in_states, h1_nodes, out_actions, batch_size: int, adam: bool):
    # TODO Test and add toggle adam
    model = NeuralNetwork([
        Conv(in_channels = 4, out_channels = 32, kernel_size = 8, stride = 4, padding = 2),
        Relu(), # Relu activation layer
        Conv(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 0),
        Relu(), # Relu activation layer
        Conv(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0),
        Relu(), # Relu activation layer
        Flatten(),
        LinearAdam(3136, 512, batch_size),   # Linear layer with adam optimizer
        Relu(), # Relu activation layer
        LinearAdam(512, out_actions, batch_size), # Linear layer with adam optimizer
    ])
    return model

def create_conv_model(in_states, h1_nodes, out_actions, batch_size: int, adam: bool):
    # TODO Test and add toggle adam
    model = NeuralNetwork([
        Conv(in_channels = 4, out_channels = 16, kernel_size = 8, stride = 4, padding = 2),
        Relu(), # Relu activation layer
        Flatten(),
        LinearAdam(400*16, 200, batch_size),   # Linear layer with adam optimizer
        Relu(), # Relu activation layer
        LinearAdam(200, out_actions, batch_size), # Linear layer with adam optimizer
    ])
    return model
        

    
        

    




   