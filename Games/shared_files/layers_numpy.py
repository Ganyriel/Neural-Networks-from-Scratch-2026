import numpy as cp
#import cupy as cp

from utils_numpy import xavier_initialization, simple_initialization


# ---------------------------------------------------------
# Linear layer
# ---------------------------------------------------------
class Linear:
    """ A fully connected linear layer implemented with NumPy arrays
        Uses: 
            - weight initializor
            - optimizer         
    """

    def __init__(
        self, in_features: int, out_features: int, batch_size: int, weight_initializor
    ) -> None:
        super(Linear, self).__init__()
        self.name = "Linear Layer"
        self.trainable = True

        self.batch_size = batch_size
        self.rng = cp.random.default_rng(seed=42)
        self.weight, self.bias = weight_initializor(in_features, out_features)
        self.grad_weight = cp.zeros((in_features, out_features))
        self.grad_bias = cp.zeros(out_features)
        self.input = cp.zeros((batch_size, in_features))

    def forward(self, inp: cp.ndarray) -> cp.ndarray:

        # If we encounter a flattened batch. Might cause some overhead
        if(inp.shape[-1] != self.weight.shape[0]):
            inp = inp.reshape(self.batch_size, -1)
        
        self.input = inp
        output = inp @ self.weight + self.bias
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        
        # Reshape in case the input is flattened
        grad_output = grad_output.reshape(-1, cp.transpose(self.weight).shape[0])  
        
        grad_input = grad_output @ cp.transpose(self.weight)
        self.grad_weight = cp.transpose(self.input) @ grad_output
        self.grad_bias = cp.sum(grad_output, axis=0)
        return grad_input

    def update(self, lr, change) -> None:
        self.weight = self.weight - lr * change[0]
        self.bias = self.bias - lr * change[1]

    def get_weights(self):
        # returns the weights and bias
        return [self.weight, self.bias]

    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.weight = new_weights[0]
        self.bias = new_weights[1]

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
    
    def get_gradients(self):
        return (self.grad_weight, self.grad_bias)


class Conv:
    """ A fully connected convolutional layer implemented with NumPy arrays
        Uses: 
            - weight initializor
            - optimizer         
    """
    # TODO RENAME VARIABLES
    # TODO ADD INITIALIZOR
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.name = "Convolutional Layer"
        self.trainable = True

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.weight = 0.1 * cp.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias= cp.zeros((out_channels, 1))

    def forward(self, inp: cp.ndarray) -> cp.ndarray:

        # If we do not provide a batch but only one input
        if len(inp.shape) == 3:
            inp = cp.array([inp])

        self.inp = inp
        batch_size, _, height, width = inp.shape

        # Padding the input
        self.padded_inp = cp.pad(inp, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')


        # Output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # STRIDE CHECK
        if int(out_height) != out_height or int(out_width) != out_width:
        
            raise  ValueError(f'Stride {self.stride} is incorrect')
        

        self.output = cp.zeros((batch_size, self.out_channels, out_height, out_width))

        # Convolution operation
        for i in range(out_height):
            for j in range(out_width):
                
                
                region = self.padded_inp[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]

                # if(region.shape[0] != 1):
                #     print("self.weight.shape: ", self.weight.shape)
                #     print("region.shape: ", region.shape)
                #     print("inp.shape: ", inp.shape)

                self.output[:, :, i, j] = cp.tensordot(region, self.weight, axes=([1, 2, 3], [1, 2, 3])) + self.bias.T

        return self.output

    def backward(self, grad_output: cp.ndarray) :

        """Backward pass for Conv2D layer."""
        batch_size, _,_,_ = self.inp.shape
        _, _, out_height, out_width = grad_output.shape

        # Initialize gradients
        self.grad_weight = cp.zeros_like(self.weight)
        self.grad_bias = cp.sum(grad_output, axis=(0, 2, 3), keepdims=True).reshape(self.out_channels, 1)
        grad_input = cp.zeros_like(self.padded_inp)
        grad_input = grad_input.astype(cp.float32)

        # Gradient computation for weights and input
        for i in range(out_height):
            for j in range(out_width):
                region = self.padded_inp[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                self.grad_weight += cp.tensordot(grad_output[:, :, i, j], region, axes=([0], [0]))  # Compute weight gradient
                for n in range(batch_size):
                    grad_input[n, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += cp.tensordot(self.weight, grad_output[n, :, i, j], axes=(0, 0))
                    

        # Remove padding if applied
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return grad_input

    def update(self, lr, change) -> None:
        self.weight = self.weight - lr * change[0]
        self.bias = self.bias - lr * change[1]

    def get_weights(self):
        # returns the weights and bias
        return [self.weight, self.bias]

    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.weight = new_weights[0]
        self.bias= new_weights[1]

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable

    def get_gradients(self):
        return (self.grad_weight, self.grad_bias)


# ---------------------------------------------------------
# Sigmoid activation
# ---------------------------------------------------------
class Sigmoid:
    """Sigmoid activation function"""

    def __init__(self) -> None:
        super(Sigmoid, self).__init__()
        self.name = "Sigmoid Layer"
        self.trainable = False
        self.input = 0
        self.output = 0

    def forward(self, inp: cp.ndarray) -> cp.ndarray:
        self.input = inp
        output = 1 / (1 + cp.exp(-self.input))
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output * (self.output * (1 - self.output))
        return grad_input

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable

# ---------------------------------------------------------
# ReLU activation
# ---------------------------------------------------------
class Relu:
    """ReLU activation function"""

    def __init__(self) -> None:
        super(Relu, self).__init__()
        self.name = "ReLu Layer"
        self.trainable = False
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.maximum(input,0)
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        # Computes the gradient of ReLU
        grad_input = grad_output.copy()
        grad_input[self.input <=0] = 0
        return grad_input

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable



# ---------------------------------------------------------
# Flatten
# ---------------------------------------------------------    
class Flatten:
    def __init__(self):
        self.name = "Flatten"
        self.trainable = False

    def forward(self, inp):
        self.inp = inp

        return inp.flatten()

    def backward(self, gradient):
        return cp.reshape(gradient, self.inp.shape).astype(cp.float64)

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable