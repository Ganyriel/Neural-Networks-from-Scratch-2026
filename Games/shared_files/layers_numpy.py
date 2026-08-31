import numpy as cp
#import cupy as cp
import time

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
        self, in_features: int, out_features: int, batch_size: int
    ) -> None:
        super(Linear, self).__init__()
        self.name = "Linear Layer"
        self.trainable = True

        self.batch_size = batch_size
        self.rng = cp.random.default_rng(seed=42)
        self.weight = None
        self.bias = None
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

    def update(self, lr, difference) -> None:
        self.weight = self.weight - lr * difference[0]
        self.bias = self.bias - lr * difference[1]
        # Debug log for vanishing and exploding gradient. Maybe
        # if (cp.max(cp.absolute(difference[0])) > 10 or cp.max(cp.absolute(difference[0])) < 0.001) or (cp.max(cp.absolute(difference[1])) > 10 or cp.max(cp.absolute(difference[1])) < 0.001):
        #     print("Difference max: ", cp.max(cp.absolute(difference[0]), cp.max(cp.absolute(difference[1]))

    def get_weights(self):
        # returns the weights and bias
        return [self.weight, self.bias]

    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.weight = new_weights[0]
        self.bias = new_weights[1]

    def print_name(self):
        print(self.name)

    def get_name(self):
        return self.name
    
    def get_size(self):
        return self.grad_weight.shape

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
        rng = cp.random.default_rng(seed = 42)
        self.weight = 0.1 * rng.standard_normal(size = (out_channels, in_channels, kernel_size, kernel_size))
        self.bias= cp.zeros((out_channels, 1)) #0.1 * rng.standard_normal(size = (out_channels, 1)) 

        # Initialize gradients
        self.grad_weight = None
        self.grad_bias = None


        # FROM THE INTERNET
        
        # temporary variables in backward()  
        self.col = None
        self.col_weights = None

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

        # Check if stride is set correctly
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
        col_weights = self.weight.reshape(self.out_channels, -1).T
        out = cp.dot(col, col_weights) + self.bias.T
        out = out.reshape(batch_size, out_height, out_width, -1).transpose(0, 3, 1, 2)

        self.col = col
        self.col_weights = col_weights

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
        dcol = cp.dot(dout, self.col_weights.T)
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

    def update(self, lr, difference) -> None:
        self.weight = self.weight - lr * difference[0]
        self.bias = self.bias - lr * difference[1]

    def get_weights(self):
        # returns the weights and bias
        return [self.weight, self.bias]

    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.weight = new_weights[0]
        self.bias= new_weights[1]

    def print_name(self):
        print(self.name)

    def get_name(self):
        return self.name

    def is_trainable(self):
        return self.trainable

    def get_gradients(self):
        return (self.grad_weight, self.grad_bias)

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

"""
# TODO implement maybe
# ---------------------------------------------------------
# Softmax
# ---------------------------------------------------------
class Softmax:
    def __init__(self) -> None:
        super(Softmax, self).__init__()
        self.name = "Softmax Layer"
        self.trainable = False
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.exp(input) / cp.sum(cp.exp(input), axis=1, keepdims=True)
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        # Computes the gradient of ReLU
        grad_input = grad_output.copy()
        raise NotImplementedError
        return grad_input

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable

    # FROM THE INTERNET
    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        exp_values = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        self.out = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.out

    def backward(self, up_grad: np.ndarray) -> np.ndarray:
        #Backward pass for Softmax using the Jacobian matrix.
        down_grad = np.empty_like(up_grad)
        for i in range(up_grad.shape[0]):
            single_output = self.out[i].reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            down_grad[i] = np.dot(jacobian, up_grad[i])
        return down_grad
"""

# ---------------------------------------------------------
# Activation functions
# ---------------------------------------------------------

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

# TODO CHECK IF CORRECT
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
        grad_input = grad_output * cp.multiply(self.output, (1 - self.output))
        return grad_input

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
    
# ---------------------------------------------------------
# Sigmoid2 activation
# ---------------------------------------------------------
class Sigmoid2:
    """Sigmoid2 activation function"""

    def __init__(self) -> None:
        super(Sigmoid2, self).__init__()
        self.name = "Sigmoid2 Layer"
        self.trainable = False
        self.input = 0
        self.output = 0

    def forward(self, inp: cp.ndarray) -> cp.ndarray:
        self.input = inp
        output = self.input / (1 + cp.exp(-self.input))
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output*((self.input * cp.sinh(self.input))/(4*cp.cosh(self.input/2)**2)+1/2)
        return grad_input

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable


# ---------------------------------------------------------
# Tanh activation
# ---------------------------------------------------------
class Tanh:
    """Tanh activation function"""

    def __init__(self) -> None:
        super(Tanh, self).__init__()
        self.name = "Tanh Layer"
        self.trainable = False
        self.input = None
        self.output = None

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.tanh(input)
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output*(1-self.output**2)
        return grad_input

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable


# ---------------------------------------------------------
# Leaky ReLU activation
# ---------------------------------------------------------
class LeakyRelu:
    """LeakyReLU activation function"""

    def __init__(self) -> None:
        super(LeakyRelu, self).__init__()
        self.name = "LeakyReLu Layer"
        self.trainable = False
        self.alpha = 0.1
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = input.copy()
        output[output<0] *= self.alpha
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output.copy()
        grad_input[self.input <=0] *= self.alpha
        return grad_input

    

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable


# TODO check if works
# ---------------------------------------------------------
# Parametric ReLU activation
# ---------------------------------------------------------
class Prelu:
    #Parametric ReLU activation function

    def __init__(self) -> None:
        super(Prelu, self).__init__()
        self.name = "PReLu Layer"
        self.trainable = True
        self.alpha = 0.25
        self.input = 0
        self.output = 0
        self.grad_alpha = None

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.maximum(input,0)+self.alpha*cp.minimum(0,input)
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output.copy()
        grad_input[self.input <=0] *= self.alpha
        one_diff = cp.minimum(0,self.input)
        self.grad_alpha = cp.sum(cp.sum(one_diff, axis=0))/one_diff.shape[0] 
        return grad_input

    def update(self, lr, difference) -> None:
        self.alpha -= lr * difference[0][0]
    
    def get_weights(self):
        # returns the alpha
        return [self.alpha]
    
    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.alpha = new_weights[0]
    
    def print_name(self):
        print(self.name)
    
    def get_name(self):
        return self.name

    def is_trainable(self):
        return self.trainable
    
    def get_gradients(self):
        return [cp.array([self.grad_alpha]),cp.array([0])]



# ---------------------------------------------------------
# Exponential LU activation
# ---------------------------------------------------------
class Elu:
    """Exponential LU activation function"""

    def __init__(self) -> None:
        super(Elu, self).__init__()
        self.name = "ELu Layer"
        self.trainable = False
        self.alpha = 0.1
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.maximum(input,0)+self.alpha*(cp.exp(cp.minimum(0,input))-1)
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output.copy()
        grad_input[self.input <=0] *= self.alpha*cp.exp(self.input[self.input <=0])
        return grad_input
    
    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
    

# ---------------------------------------------------------
# Sinusoid activation
# ---------------------------------------------------------
class Sinusoid:
    """Sinusoid activation function"""

    def __init__(self) -> None:
        super(Sinusoid, self).__init__()
        self.name = "Sinusoid Layer"
        self.trainable = False
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.sin(input)
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output*cp.cos(self.input)
        return grad_input

    

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
    
# ---------------------------------------------------------
# Cosinusoid activation
# ---------------------------------------------------------
class Cosinusoid:
    """Cosinusoid activation function"""

    def __init__(self) -> None:
        super(Cosinusoid, self).__init__()
        self.name = "Cosinusoid Layer"
        self.trainable = False
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.cos(input)
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output*(-cp.sin(self.input))
        return grad_input

    
    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
    
    
# TODO CHECK IF WORKS
# ---------------------------------------------------------
# Gaussian activation
# ---------------------------------------------------------
class Gaussian:
    """Gaussian activation function"""

    def __init__(self) -> None:
        super(Gaussian, self).__init__()
        self.name = "Gaussian Layer"
        self.trainable = False
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.exp(-input**2)
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output*(-2)*cp.multiply(self.input,self.output) 
        return grad_input


    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
    


# ---------------------------------------------------------
# Softplus activation
# ---------------------------------------------------------
class Softplus:
    """Softplus activation function"""

    def __init__(self) -> None:
        super(Softplus, self).__init__()
        self.name = "Softplus Layer"
        self.trainable = False
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.log(1+cp.exp(input))
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output*cp.log(1+cp.exp(-self.input))
        return grad_input


    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
    
# ---------------------------------------------------------
# Scaled Exponential LU activation
# ---------------------------------------------------------
class Selu:
    """Scaled Exponential LU activation function"""

    def __init__(self) -> None:
        super(Selu, self).__init__()
        self.name = "SeLu Layer"
        self.trainable = False
        self.l = 1.0507
        self.alpha = 1.67326
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = self.l * (cp.maximum(input,0)+self.alpha*(cp.exp(cp.minimum(0,input))-1))
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output.copy()
        grad_input[self.input <=0] *= (self.alpha*cp.exp(self.input[self.input <=0]))
        grad_input *= self.l
        return grad_input


    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
    

# ---------------------------------------------------------
# Arctanh activation
# ---------------------------------------------------------
class Atanh:
    """Arctanh activation function"""

    def __init__(self) -> None:
        super(Atanh, self).__init__()
        self.name = "Atanh Layer"
        self.trainable = False
        self.input = None
        self.output = None

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = cp.atanh(input)
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        grad_input = grad_output/(self.input**2 + 1)
        return grad_input

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
    

# ---------------------------------------------------------
# Identity activation
# ---------------------------------------------------------
class Identity:
    """Identity activation function"""

    def __init__(self) -> None:
        super(Identity, self).__init__()
        self.name = "Identity Layer"
        self.trainable = False
        self.input = 0
        self.output = 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = self.input
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        # Computes the gradient of ReLU
        grad_input = grad_output.copy()
        return grad_input

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable

# ---------------------------------------------------------
# Swish activation
# ---------------------------------------------------------
class Swish:
    """Swish activation function"""

    def __init__(self) -> None:
        super(Swish, self).__init__()
        self.name = "Swish Layer"
        self.trainable = False
        self.beta = 1.702 # default: 1.702
        self.input = None
        self.output = None

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = self.input / (1 + cp.exp(-self.beta*self.input))
        self.output = output
        return output

    def backward(self, grad_output: cp.ndarray) -> cp.ndarray:
        beta_x = self.beta * self.input
        grad_input = grad_output*((beta_x * cp.sinh(beta_x))/(4*cp.cosh(beta_x/2)**2)+1/2)
        return grad_input

    def print_name(self):
        print(self.name)

    def is_trainable(self):
        return self.trainable
