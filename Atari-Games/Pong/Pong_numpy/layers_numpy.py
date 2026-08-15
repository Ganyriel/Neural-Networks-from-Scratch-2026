import numpy as cp
#import cupy as cp

from utils_numpy import xavier_initialization, other_initialization


# ---------------------------------------------------------
# Linear layer
# ---------------------------------------------------------
class Linear:
    """ A fully connected layer implemented with NumPy arrays, without any sophistication
        Uses: 
            - simple initialization of weights
            - gradient descent          
    """

    def __init__(
        self, in_features: int, out_features: int, batch_size: int
    ) -> None:
        super(Linear, self).__init__()
        self.name = "Linear Layer"
        self.batch_size = batch_size
        self.rng = cp.random.default_rng(seed=42)
        self.weight, self.bias = xavier_initialization(in_features, out_features)
        self.grad_weight = cp.zeros((in_features, out_features))
        self.grad_bias = cp.zeros(out_features)
        self.input = cp.zeros((batch_size, in_features))

    def forward(self, inp: cp.ndarray) -> cp.ndarray:
        # TODO PROVISIONAL
        # print("self.weight.shape linear ", self.weight.shape)
        # print("input.shape linear before", input.shape)
        if(inp.shape[-1] != self.weight.shape[0]):
            inp = inp.reshape(self.batch_size, -1)
        # print("self.batch_size: ", self.batch_size)
        # print("self.weight.shape linear: ", self.weight.shape)
        # print("input.shape[-1]: ", input.shape[-1])
        # print("input.shape linear after: ", input.shape)
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

    def update(self, lr) -> None:
        self.weight = self.weight - lr * self.grad_weight / self.batch_size
        self.bias = self.bias - lr * self.grad_bias / self.batch_size

    def get_weights(self):
        # returns the weights and bias
        return [self.weight, self.bias]

    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.weight = new_weights[0]
        self.bias = new_weights[1]

    def print_name(self):
        print(self.name)


# ---------------------------------------------------------
# Linear layer with ADAM
# ---------------------------------------------------------
class LinearAdam:
    """A fully connected layer implemented with NumPy arrays."""

    def __init__(
        self, in_features: int, out_features: int, batch_size: int
    ) -> None:
        super(LinearAdam, self).__init__()
        self.name = "Linear Layer with Adam"
        self.batch_size = batch_size

        # For comparison: more primitive initialization of weights and bias
        # self.weight = cp.random.normal(size=(in_features, out_features)) * cp.sqrt(1.0 / in_features)
        # self.bias = cp.random.normal(size=(out_features,)) * cp.sqrt(1.0 / in_features)

        # Initialization of weights and bias using Xavier Initialization
        self.weight, self.bias = xavier_initialization(in_features, out_features)
        

        # Initialization for forward and backward pass
        self.grad_weight = cp.zeros((in_features, out_features))
        self.grad_bias = cp.zeros(out_features)
        self.input = cp.zeros((batch_size, in_features))


        # Attributes for ADAM

        # Shared for weights and bias
        self.e = 1e-8
        self.lr = 0.001 # default: 0.001
        self.b_1 = 0.9
        self.b_1_t = self.b_1
        self.b_2 = 0.999
        self.b_2_t = self.b_2

        # For weights
        self.m_w = 0
        self.v_w = 0
          
        # For bias
        self.m_b = 0
        self.v_b = 0
        
        
    def forward(self, inp: cp.ndarray) -> cp.ndarray:
        

        # TODO PROVISIONAL

        # Debug log
        # print("self.weight.shape linear ", self.weight.shape)
        # print("inp.shape linear before", inp.shape)
        if(inp.shape[-1] != self.weight.shape[0]):
            inp = inp.reshape(self.batch_size, -1)

        # Debug log
        # print("self.batch_size: ", self.batch_size)
        # print("self.weight.shape linear: ", self.weight.shape)
        # print("inp.shape[-1]: ", inp.shape[-1])
        # print("inp.shape linear after: ", inp.shape)

        self.input = inp
        output = inp @ self.weight + self.bias

        # Debug log
        # print("Linear out shape: ", output.shape) 
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        
        # Reshape in case the input is flattened
        grad_output = grad_output.reshape(-1, cp.transpose(self.weight).shape[0])  
        
        grad_input = grad_output @ cp.transpose(self.weight)
        self.grad_weight = cp.transpose(self.input) @ grad_output
        self.grad_bias = cp.sum(grad_output, axis=0)
        return grad_input

    def update(self, learning_rate) -> None:
        "Implementation of the ADAM optimizer"

        # Weight update
        self.m_w = self.b_1 * self.m_w +(1-self.b_1) * self.grad_weight / self.batch_size
        self.v_w = self.b_2 * self.v_w + (1-self.b_2) * (self.grad_weight / self.batch_size)**2  
        m_hat_w = self.m_w / (1-self.b_1_t) 
        v_hat_w = self.v_w / (1-self.b_2_t)
        self.weight  = self.weight - self.lr * m_hat_w / (cp.sqrt(v_hat_w) + self.e)                    

        # Bias update
        self.m_b = self.b_1 * self.m_b +(1-self.b_1) * self.grad_bias / self.batch_size
        self.v_b = self.b_2 * self.v_b + (1-self.b_2) * (self.grad_bias / self.batch_size)**2  
        m_hat_b = self.m_b / (1-self.b_1_t) 
        v_hat_b = self.v_b / (1-self.b_2_t)
        self.bias = self.bias - self.lr * m_hat_b / (cp.sqrt(v_hat_b) + self.e)   

        # Shared parameter update
        self.b_1_t = self.b_1_t*self.b_1
        self.b_2_t = self.b_2_t*self.b_2

    def get_weights(self):
        # returns the weights and bias
        return [self.weight, self.bias]

    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.weight = new_weights[0]
        self.bias = new_weights[1]

    def print_name(self):
        print(self.name)    

class Conv:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.name = "Convolutional Layer"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and biases
        self.w = 0.1 * cp.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.b = cp.zeros((out_channels, 1))

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
        

        self.out = cp.zeros((batch_size, self.out_channels, out_height, out_width))

        # Convolution operation
        for i in range(out_height):
            for j in range(out_width):
                
                
                region = self.padded_inp[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]

                # if(region.shape[0] != 1):
                #     print("self.w.shape: ", self.w.shape)
                #     print("region.shape: ", region.shape)
                #     print("inp.shape: ", inp.shape)

                self.out[:, :, i, j] = cp.tensordot(region, self.w, axes=([1, 2, 3], [1, 2, 3])) + self.b.T

        return self.out

    def backward(self, up_grad: cp.ndarray) :

        """Backward pass for Conv2D layer."""
        batch_size, _,_,_ = self.inp.shape
        _, _, out_height, out_width = up_grad.shape

        # Initialize gradients
        self.dw = cp.zeros_like(self.w)
        self.db = cp.sum(up_grad, axis=(0, 2, 3), keepdims=True).reshape(self.out_channels, 1)
        down_grad = cp.zeros_like(self.padded_inp)
        down_grad = down_grad.astype(cp.float32)

        # Gradient computation for weights and input
        for i in range(out_height):
            for j in range(out_width):
                region = self.padded_inp[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                self.dw += cp.tensordot(up_grad[:, :, i, j], region, axes=([0], [0]))  # Compute weight gradient
                for n in range(batch_size):
                    down_grad[n, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size] += cp.tensordot(self.w, up_grad[n, :, i, j], axes=(0, 0))
                    

        # Remove padding if applied
        if self.padding > 0:
            down_grad = down_grad[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return down_grad

    def update(self, lr: float) -> None:
        """Update weights and biases."""
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def get_weights(self):
        # returns the weights and bias
        return [self.w, self.b]

    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.w = new_weights[0]
        self.b = new_weights[1]

    def print_name(self):
        print(self.name)



# ---------------------------------------------------------
# Sigmoid activation
# ---------------------------------------------------------
class Sigmoid:
    """Sigmoid activation function"""

    def __init__(self) -> None:
        super(Sigmoid, self).__init__()
        self.name = "Sigmoid Layer"
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

    def update(self, learning_rate) -> None:
        pass

    def get_weights(self):
        return [1]

    def set_weights(self, new_weights):
        pass

    def print_name(self):
        print(self.name)

# ---------------------------------------------------------
# ReLU activation
# ---------------------------------------------------------
class Relu:
    """ReLU activation function"""

    def __init__(self) -> None:
        super(Relu, self).__init__()
        self.name = "ReLu Layer"
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

    def update(self, learning_rate) -> None:
        pass

    def get_weights(self):
        return [1]

    def set_weights(self, new_weights):
        pass

    def print_name(self):
        print(self.name)



# ---------------------------------------------------------
# Flatten
# ---------------------------------------------------------    
class Flatten:
    def __init__(self):
        self.name = "Flatten"
        self.trainable = None

    def forward(self, inp):
        self.inp = inp

        return inp.flatten()

    def backward(self, gradient):
        return cp.reshape(gradient, self.inp.shape).astype(cp.float64)

    def update(self, learning_rate) -> None:
        pass

    def get_weights(self):
        return [1]

    def set_weights(self, new_weights):
        pass

    def print_name(self):
        print(self.name)