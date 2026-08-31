import numpy as cp
#import cupy as cp
from layers_numpy import Linear, Relu, Conv, Flatten
from utils_numpy import xavier_initialization, Adam

class NeuralNetwork:
    def __init__(self, layers, optimizer, batch_size, weight_initializor, in_states, out_actions):
        self.layers = layers
        self.trainable_layers = [layer for layer in self.layers if layer.is_trainable()]
        self.batch_size = batch_size
        self.optimizer = optimizer(self.trainable_layers, self.batch_size)
        self.weight_initializator = weight_initializor

        # Perform weight initialization
        # TODO ADD WEIGHT INITIALIZATION FOR CONVOLUTIONAL LAYERS
        for layer in self.trainable_layers:
            if(layer.get_name() == "Linear Layer"):
                layer.set_weights(weight_initializor(layer.get_size()[0], layer.get_size()[1], in_states = in_states, out_actions = out_actions))

        
    def forward(self, x: cp.ndarray) -> cp.ndarray:
        # Preparing input
        x = cp.array(x)

        # Performing forward pass for every layer
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x: cp.ndarray) -> None:
        #Performing backward pass for every layer
        for layer in reversed(self.layers):
            x = layer.backward(x)
    
    def update(self) -> None:
        # Using optimizer to get the learning rate and the value 
        change = self.optimizer.step(self.get_gradients())        
        for i in range(len(self.trainable_layers)):
            self.trainable_layers[i].update(change[0],change[1][i])

    def get_weights(self):
        # Returns weights of the neurons
        return [layer.get_weights() for layer in self.trainable_layers]
        
    def set_weights(self, weights):
        # Sets weights of the neurons
        for i in range(len(self.trainable_layers)):
            self.trainable_layers[i].set_weights(weights[i])

    def print_name(self):
        # Prints the name of every layer
        print("")
        print("Network architecture: ")
        for layer in self.layers:
            layer.print_name()
        print("")
        print("Weight initializator: ", str(self.weight_initializator.__name__))
        print("Optimizer: ", str(self.optimizer.get_name()))
        print("")

    def get_gradients(self):
        return [layer.get_gradients() for layer in self.trainable_layers] 
        
def create_network(in_states, h1_nodes, out_actions, batch_size: int, model_name = "three_layers", weight_initializor = xavier_initialization, optimizer = Adam, activation_function = Relu):
    if(model_name == "three_layers"):
        model = NeuralNetwork([
            Flatten(), # Flattening Input
            Linear(in_states, h1_nodes, batch_size),   # Linear layer 
            activation_function(), # Activation function
            Linear(h1_nodes, h1_nodes, batch_size),   # Linear layer 
            activation_function(), # Activation function
            Linear(h1_nodes, out_actions, batch_size), # Linear layer 
        ],optimizer=optimizer, batch_size=batch_size, weight_initializor = weight_initializor, in_states = in_states, out_actions = out_actions)

    if(model_name == "two_layers"):
        model = NeuralNetwork([
            Flatten(), # Flattening Input
            Linear(in_states, h1_nodes, batch_size),   # Linear layer with adam optimizer
            activation_function(), # Activation function
            Linear(h1_nodes, out_actions, batch_size), # Linear layer with adam optimizer
        ],optimizer=optimizer, batch_size=batch_size, weight_initializor = weight_initializor, in_states = in_states, out_actions = out_actions)
        
    if(model_name == "one_layer"):
        model = NeuralNetwork([
            Flatten(), # Flattening Input
            Linear(in_states, out_actions, batch_size)  # Linear layer with adam optimizer
        ],optimizer=optimizer, batch_size=batch_size, weight_initializor = weight_initializor, in_states = in_states, out_actions = out_actions)

    if(model_name == "triple_convolutional_model_pong"):
        # TODO Test 
        model = NeuralNetwork([
            Conv(in_channels = 4, out_channels = 32, kernel_size = 8, stride = 4, padding = 2), # Convolutional layer
            activation_function(), # Activation function
            Conv(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2, padding = 0), # Convolutional layer
            activation_function(), # Activation function
            Conv(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 0), # Convolutional layer
            activation_function(), # Activation function
            Flatten(), # Flattening Input
            Linear(3136, 512, batch_size),   # Linear layer with adam optimizer
            activation_function(), # Activation function
            Linear(512, out_actions, batch_size), # Linear layer with adam optimizer
        ],optimizer=optimizer, batch_size=batch_size, weight_initializor = weight_initializor, in_states = in_states, out_actions = out_actions)

    if(model_name == "convolutional_pong"):
        # TODO Test 
        model = NeuralNetwork([
            Conv(in_channels = 4, out_channels = 16, kernel_size = 8, stride = 4, padding = 2), # Convolutional layer
            activation_function(), # Activation function
            Flatten(), # Flattening Input
            Linear(400*16, 200, batch_size),   # Linear layer with adam optimizer
            activation_function(), # Activation function
            Linear(200, out_actions, batch_size), # Linear layer with adam optimizer
        ],optimizer=optimizer, batch_size=batch_size, weight_initializor = weight_initializor, in_states = in_states, out_actions = out_actions)

    return model



    




        

    
        

    




   