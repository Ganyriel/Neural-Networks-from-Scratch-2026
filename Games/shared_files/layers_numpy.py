import numpy as cp
#import cupy as cp
import time

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

        # Initialize gradients
        self.grad_weight = None
        self.grad_bias = None


        # FROM THE INTERNET
        
        # temporary variables in backward()  
        self.col = None
        self.col_W = None

        # TODO REMOVE WHEN CERTAIN WHICH METHOD TO USE
        # gradient of weight and bias 
        self.dW = None
        self.db = None


    # FROM THE INTERNET
    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        """
        Parameters
        ----------
        input_ Data: input data composed of 4-dimensional arrays (data volume, channel, height and length)
        filter_ H: filter high
        filter_ W: length of filter
        Stripe: stride
        Pad: fill
        Returns
        -------
        Col: 2-dimensional array
        """
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1
        img = cp.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = cp.zeros((N, C, filter_h, filter_w, out_h, out_w))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col
    
    # FROM THE INTERNET
    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        """
        Parameters
        ----------
        col :
        input_ Shape: the shape of the input data (for example: (10, 1, 28, 28))
        filter_h :
        filter_w
        stride
        pad
        Returns
        -------
        """
        N, C, H, W = input_shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
        img = cp.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
        return img[:, :, pad:H + pad, pad:W + pad]


    def forward(self, inp: cp.ndarray) -> cp.ndarray:
        # If we do not provide a batch but only one input
        if len(inp.shape) == 3:
            inp = cp.array([inp])

        self.inp = inp
        batch_size, _, height, width = inp.shape

        """
        # Part of old forward and backward pass
        # Padding the input
        self.padded_inp = cp.pad(inp, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        # Old part end
        """


        # Output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # STRIDE CHECK
        if int(out_height) != out_height or int(out_width) != out_width:
        
            raise  ValueError(f'Stride {self.stride} is incorrect')
        


        """
        # Part of old forward pass
        self.output = cp.zeros((batch_size, self.out_channels, out_height, out_width))

        # Convolution operation
        for i in range(out_height):
            for j in range(out_width):
                region = self.padded_inp[:, :, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size]
                self.output[:, :, i, j] = cp.tensordot(region, self.weight, axes=([1, 2, 3], [1, 2, 3])) + self.bias.T
        # Old part end
        """
        
        # FROM THE INTERNET
        FH = self.kernel_size
        FW = self.kernel_size

        col = self.im2col(input_data = inp, filter_h = FH, filter_w = FW, stride=self.stride, pad = self.padding)
        col_W = self.weight.reshape(self.out_channels, -1).T
        out = cp.dot(col, col_W) + self.bias.T
        out = out.reshape(batch_size, out_height, out_width, -1).transpose(0, 3, 1, 2)

        self.col = col
        self.col_W = col_W

        self.output = out

        
        # Debug log
        # print("Forward difference: ", cp.max(cp.abs(out - self.output)))
        # print("Forward minimum: ", cp.min(cp.abs(out)))
        # print("Forward maximum: ", cp.max(cp.abs(out)))
        # print("")
        
        
        

        return self.output


    def backward(self, grad_output: cp.ndarray) :
        """Backward pass for Conv2D layer."""
        
        
        """
        # Part of old forward and backward pass
        # Initializing weight gradient
        self.grad_weight = cp.zeros_like(self.weight)

        # Calculation of bias gradient
        self.grad_bias = cp.sum(grad_output, axis=(0, 2, 3), keepdims=True).reshape(self.out_channels, 1)

        batch_size, _,_,_ = self.inp.shape
        _, _, out_height, out_width = grad_output.shape
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

        # Old part end
        """

        
        # FROM THE INTERNET
        FH = self.kernel_size
        FW = self.kernel_size
        
        dout = grad_output.transpose(0,2,3,1).reshape(-1, self.out_channels)

        # Calculation of bias gradient
        self.db = cp.sum(dout, axis=0).reshape(-1,1)
        
        # Calculation of weight gradient
        self.dW = cp.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(self.out_channels, self.in_channels, FH, FW)

        # Calculation of backward gradient
        dcol = cp.dot(dout, self.col_W.T)
        dx = self.col2im(dcol, self.inp.shape, FH, FW, self.stride, self.padding)
        
        # Debug log
        # print("Update difference (weights): ", cp.max(cp.abs(self.dW - self.grad_weight)))
        # print("Update minimum (weights): ", cp.min(cp.abs(self.dW)))
        # print("")
        # print("Update difference (bias): ", cp.max(cp.abs(self.db - self.grad_bias)))
        # print("Update minimum (bias): ", cp.min(cp.abs(self.db)))
        # print("Update maximum(bias): ", cp.max(cp.abs(self.db)))
        # print("Error percentage (minimum): ", cp.max(cp.abs(self.db - self.grad_bias))/cp.min(cp.abs(self.db))*100, "%")
        # print("Error percentage (maximum): ", cp.max(cp.abs(self.db - self.grad_bias))/cp.max(cp.abs(self.db))*100, "%")
        # print("")

        # print("Backward difference: ", cp.max(cp.abs(dx - grad_input)))
        # print("Backward minimum: ", cp.min(cp.abs(dx)))
        # print("")

        # TODO REMOVE WHEN CERTAIN WHICH METHOD TO USE
        self.grad_weight = self.dW
        self.grad_bias = self.db
        grad_input = dx
        
        
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