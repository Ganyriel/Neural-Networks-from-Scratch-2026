import numpy as np

# ---------------------------------------------------------
# Linear layer
# ---------------------------------------------------------
class Linear:
    """ A fully connected linear layer implemented with NumPy arrays
        Uses: 
            - weight initializor
            - optimizer         
    """

    def __init__(self, in_features: int, out_features: int, batch_size: int) -> None:
        # Initialization
        super(Linear, self).__init__()
        self.name = "Linear Layer"
        self.trainable = True
        self.batch_size = batch_size


        # Initializing weights and bias
        self.weight = None
        self.bias = None


        # Initializing variables used in backward pass and weight and bias update
        self.grad_weight = np.zeros((in_features, out_features))
        self.grad_bias = np.zeros(out_features)
        self.input = np.zeros((batch_size, in_features))

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Reshaping in case we encounter a batch
        if(inp.shape[-1] != self.weight.shape[0]):
            inp = inp.reshape(self.batch_size, -1)

        # Saving input for backward pass
        self.input = inp

        # Calculating the output
        output = inp @ self.weight + self.bias

        # Returning the output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass
        
        # Reshape in case the input is flattened
        grad_output = grad_output.reshape(-1, np.transpose(self.weight).shape[0])  

        # Calculate the gradient passed downwards
        grad_input = grad_output @ np.transpose(self.weight)

        # Calculate the gradient of the weights and bias
        self.grad_weight = np.transpose(self.input) @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0)

        # Return downwards gradient
        return grad_input

    def update(self, lr, difference) -> None:
        # Performs weight update
        self.weight = self.weight - lr * difference[0]
        self.bias = self.bias - lr * difference[1]

    def get_weights(self):
        # Returns weights and bias
        return [self.weight, self.bias]

    def set_weights(self, new_weights):
        # Sets weights and bias 
        self.weight = new_weights[0]
        self.bias = new_weights[1]

    def print_name(self):
        # Prints name
        print(self.name)

    def get_name(self):
        # Returns name
        return self.name
    
    def get_size(self):
        # Returns layer size
        return self.grad_weight.shape

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    
    def get_gradients(self):
        # Returns the gradients calculated in backward
        return (self.grad_weight, self.grad_bias)


class Conv:
    """ A fully connected convolutional layer implemented with NumPy arrays
        Uses: 
            - optimizer         
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        # Initialization
        super().__init__()
        self.name = "Convolutional Layer"
        self.trainable = True

        # Saving input parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights and bias
        rng = np.random.default_rng(seed = 42)
        self.weight = 0.1 * rng.standard_normal(size = (out_channels, in_channels, kernel_size, kernel_size))
        self.bias= 0.1 * rng.standard_normal(size = (out_channels, 1)) 

        # Initialize gradients
        self.grad_weight = None
        self.grad_bias = None

        # Temporary variables used backward  
        self.col = None
        self.col_weights = None

    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        """Extracts columns from image"""

        # Getting image size
        N, C, H, W = input_data.shape

        # Calculating image size after convolution
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        # Applay padding on image
        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

        # Initialize columns
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        # Extraction
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        # Reshape columns into right shape
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

        # Return columns
        return col
    

    def col2im(self, col, inp_shape, filter_h, filter_w, stride=1, pad=0):
        """Extracts back columns into image"""

        # Getting image size
        N, C, H, W = inp_shape

        # Calculating image size after convolution
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        # Reshape columns into right shape
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        # Initialize image
        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

        # Extraction
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        # Return original image
        return img[:, :, pad:H + pad, pad:W + pad]


    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # If we do not provide a batch but only one inp
        if len(inp.shape) == 3:
            inp = np.array([inp])

        # Saving input for backwards pass
        self.input = inp
        batch_size, _, height, width = inp.shape

        # Output dimensions
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Check if stride is set correctly
        if int(out_height) != out_height or int(out_width) != out_width:
            raise  ValueError(f'Stride {self.stride} is incorrect')
        
        # Set filter height and width
        FH = self.kernel_size
        FW = self.kernel_size

        # Calculate column and column of weights
        col = self.im2col(input_data = inp, filter_h = FH, filter_w = FW, stride=self.stride, pad = self.padding)
        col_weights = self.weight.reshape(self.out_channels, -1).T

        # Perform convolution and reshape the output again
        out = np.dot(col, col_weights) + self.bias.T
        out = out.reshape(batch_size, out_height, out_width, -1).transpose(0, 3, 1, 2)

        # Set column and column of weights
        self.col = col
        self.col_weights = col_weights

        # Return output
        return out


    def backward(self, grad_output: np.ndarray) :
        """Backward pass for Conv2D layer."""
        
        # Set filter height and width
        FH = self.kernel_size
        FW = self.kernel_size

        # Reshape output gradient as necessary
        dout = grad_output.transpose(0,2,3,1).reshape(-1, self.out_channels)

        # Calculation of bias gradient
        self.grad_bias = np.sum(dout, axis=0).reshape(-1,1)
        
        # Calculation of weight gradient
        self.grad_weight = np.dot(self.col.T, dout)
        self.grad_weight = self.grad_weight.transpose(1, 0).reshape(self.out_channels, self.in_channels, FH, FW)

        # Calculation of backward gradient
        dcol = np.dot(dout, self.col_weights.T)
        grad_input = self.col2im(dcol, self.input.shape, FH, FW, self.stride, self.padding)
        
        # Return downwards gradient
        return grad_input

    def update(self, lr, difference) -> None:
        # Performs weight update
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
        # Prints name
        print(self.name)

    def get_name(self):
        # Returns name
        return self.name

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable

    def get_gradients(self):
        # Returns the gradients calculated in backward
        return (self.grad_weight, self.grad_bias)

# ---------------------------------------------------------
# Flatten
# ---------------------------------------------------------    
class Flatten:
    """Layer for flattening input"""

    def __init__(self):
        # Initializing
        self.name = "Flatten"
        self.trainable = False

    def forward(self, inp):
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Return flattened input
        return inp.flatten()

    def backward(self, gradient):
        # Performs a backward pass

        # Unflattens gradient
        return np.reshape(gradient, self.input.shape).astype(np.float64)

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable


# ---------------------------------------------------------
# Activation functions
# ---------------------------------------------------------

# ---------------------------------------------------------
# ReLU activation
# ---------------------------------------------------------
class Relu:
    """ReLU activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Relu, self).__init__()
        self.name = "ReLu Layer"
        self.trainable = False

        self.input = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # ReLU calculation
        output = np.maximum(inp,0)

        # Return output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of ReLU multiplied with the previous gradient
        grad_input = grad_output.copy()
        grad_input[self.input <=0] = 0

        # Return downwards gradient
        return grad_input

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable


# ---------------------------------------------------------
# Sigmoid activation
# ---------------------------------------------------------
class Sigmoid:
    """Sigmoid activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Sigmoid, self).__init__()
        self.name = "Sigmoid Layer"
        self.trainable = False

        self.input = 0
        self.output = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Sigmoid calculation
        output = 1 / (1 + np.exp(-self.input))

        # Saving output for backward pass
        self.output = output

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Sigmoid multiplied with the previous gradient
        grad_input = grad_output * np.multiply(self.output, (1 - self.output))

        # Return downwards gradient
        return grad_input

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    
# ---------------------------------------------------------
# Sigmoid2 activation
# ---------------------------------------------------------
class Sigmoid2:
    """Different version of the Sigmoid activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Sigmoid2, self).__init__()
        self.name = "Sigmoid2 Layer"
        self.trainable = False

        self.input = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Sigmoid calculation
        output = self.input / (1 + np.exp(-self.input))

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Sigmoid multiplied with the previous gradient
        grad_input = grad_output*((self.input * np.sinh(self.input))/(4*np.cosh(self.input/2)**2)+1/2)

        # Return downwards gradient
        return grad_input

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable


# ---------------------------------------------------------
# Tanh activation
# ---------------------------------------------------------
class Tanh:
    """Tanh activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Tanh, self).__init__()
        self.name = "Tanh Layer"
        self.trainable = False
        self.input = None
        self.output = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Tanh calculation
        output = np.tanh(inp)

        # Saving output for backward pass
        self.output = output

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Tanh multiplied with the previous gradient
        grad_input = grad_output*(1-self.output**2)

        # Return downwards gradient
        return grad_input

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable


# ---------------------------------------------------------
# Leaky ReLU activation
# ---------------------------------------------------------
class LeakyRelu:
    """Leaky ReLU activation function"""

    def __init__(self) -> None:
        # Initializing
        super(LeakyRelu, self).__init__()
        self.name = "LeakyReLu Layer"
        self.trainable = False

        # Parameter for Leaky ReLU 
        self.alpha = 0.1

        self.input = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Leaky ReLU calculation
        output = inp.copy()
        output[output<0] *= self.alpha

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Leaky Relu multiplied with the previous gradient
        grad_input = grad_output.copy()
        grad_input[self.input <=0] *= self.alpha

        # Return downwards gradient
        return grad_input

    

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable

# ---------------------------------------------------------
# Parametric ReLU activation
# ---------------------------------------------------------
class Prelu:
    #Parametric ReLU activation function

    def __init__(self) -> None:
        # Initializing
        super(Prelu, self).__init__()
        self.name = "PReLu Layer"
        self.trainable = True
        self.alpha = 0.25
        self.input = 0
        self.grad_alpha = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # PReLU calculation
        output = np.maximum(inp,0)+self.alpha*np.minimum(0,inp)

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Parametrized ReLU multiplied with the previous gradient
        grad_input = grad_output.copy()
        grad_input[self.input <=0] *= self.alpha

        # Computes the gradient for weight alpha
        one_diff = np.minimum(0,self.input)
        self.grad_alpha = np.sum(np.sum(one_diff, axis=0))/one_diff.shape[0] 

        # Return downwards gradient
        return grad_input

    def update(self, lr, difference) -> None:
        # Performs weight update
        self.alpha -= lr * difference[0][0]
    
    def get_weights(self):
        # returns the alpha
        return [self.alpha]
    
    def set_weights(self, new_weights):
        # sets weights and bias of the neural network
        self.alpha = new_weights[0]
    
    def print_name(self):
        # Prints name
        print(self.name)
    
    def get_name(self):
        # Returns name
        return self.name

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    
    def get_gradients(self):
        # Returns the gradients calculated in backward (there is no bias)
        return [np.array([self.grad_alpha]),np.array([0])]



# ---------------------------------------------------------
# Exponential LU activation
# ---------------------------------------------------------
class Elu:
    """Exponential LU activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Elu, self).__init__()
        self.name = "ELu Layer"
        self.trainable = False
        self.alpha = 0.1
        self.input = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Exponential LU calculation
        output = np.maximum(inp,0)+self.alpha*(np.exp(np.minimum(0,inp))-1)

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Exponential LU multiplied with the previous gradient
        grad_input = grad_output.copy()
        grad_input[self.input <=0] *= self.alpha*np.exp(self.input[self.input <=0])

        # Return downwards gradient
        return grad_input
    
    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    

# ---------------------------------------------------------
# Sinusoid activation
# ---------------------------------------------------------
class Sinusoid:
    """Sinusoid activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Sinusoid, self).__init__()
        self.name = "Sinusoid Layer"
        self.trainable = False
        self.input = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Sinus calculation
        output = np.sin(inp)

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Sinus multiplied with the previous gradient
        grad_input = grad_output*np.cos(self.input)


        # Return downwards gradient
        return grad_input

    

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    
# ---------------------------------------------------------
# Cosinusoid activation
# ---------------------------------------------------------
class Cosinusoid:
    """Cosinusoid activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Cosinusoid, self).__init__()
        self.name = "Cosinusoid Layer"
        self.trainable = False
        self.input = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Cosinus calculation
        output = np.cos(inp)

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Cosinus multiplied with the previous gradient
        grad_input = grad_output*(-np.sin(self.input))

        # Return downwards gradient
        return grad_input

    
    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    
# ---------------------------------------------------------
# Gaussian activation
# ---------------------------------------------------------
class Gaussian:
    """Gaussian activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Gaussian, self).__init__()
        self.name = "Gaussian Layer"
        self.trainable = False

        self.input = 0
        self.output = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Gaussian calculation
        output = np.exp(-inp**2)

        # Saving output for backward pass
        self.output = output

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of SGaussian multiplied with the previous gradient
        grad_input = grad_output*(-2)*np.multiply(self.input,self.output) 

        # Return downwards gradient
        return grad_input


    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    


# ---------------------------------------------------------
# Softplus activation
# ---------------------------------------------------------
class Softplus:
    """Softplus activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Softplus, self).__init__()
        self.name = "Softplus Layer"

        self.trainable = False
        self.input = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Softplus calculation
        output = np.log(1+np.exp(inp))

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Softplus multiplied with the previous gradient
        grad_input = grad_output*np.log(1+np.exp(-self.input))

        # Return downwards gradient
        return grad_input


    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    
# ---------------------------------------------------------
# Scaled Exponential LU activation
# ---------------------------------------------------------
class Selu:
    """Scaled Exponential LU activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Selu, self).__init__()
        self.name = "SeLu Layer"
        self.trainable = False

        self.l = 1.0507
        self.alpha = 1.67326

        self.input = 0

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Scaled Exponential LU calculation
        output = self.l * (np.maximum(inp,0)+self.alpha*(np.exp(np.minimum(0,inp))-1))

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Scaled Exponential LU multiplied with the previous gradient
        grad_input = grad_output.copy()
        grad_input[self.input <=0] *= (self.alpha*np.exp(self.input[self.input <=0]))
        grad_input *= self.l

        # Return downwards gradient
        return grad_input


    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    

# ---------------------------------------------------------
# Arctanh activation
# ---------------------------------------------------------
class Atanh:
    """Arctanh activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Atanh, self).__init__()
        self.name = "Atanh Layer"
        self.trainable = False

        self.input = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Atanh calculation
        output = np.atanh(inp)

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Atanh multiplied with the previous gradient
        grad_input = grad_output/(self.input**2 + 1)

        # Return downwards gradient
        return grad_input

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
    

# ---------------------------------------------------------
# Identity activation
# ---------------------------------------------------------
class Identity:
    """Identity activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Identity, self).__init__()
        self.name = "Identity Layer"
        self.trainable = False

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Returns input
        return inp

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass
       
        # Return downwards gradient
        return grad_output.copy()

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable

# ---------------------------------------------------------
# Swish activation
# ---------------------------------------------------------
class Swish:
    """Swish activation function"""

    def __init__(self) -> None:
        # Initializing
        super(Swish, self).__init__()
        self.name = "Swish Layer"
        self.trainable = False
        self.beta = 1.702 # default: 1.702
        self.input = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # Forward operation of layer

        # Saving input for backwards pass
        self.input = inp

        # Swish calculation
        output = self.input / (1 + np.exp(-self.beta*self.input))

        # Returns output
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Performs a backward pass

        # Computes the gradient of Swish multiplied with the previous gradient
        beta_x = self.beta * self.input
        grad_input = grad_output*((beta_x * np.sinh(beta_x))/(4*np.cosh(beta_x/2)**2)+1/2)

        # Return downwards gradient
        return grad_input

    def print_name(self):
        # Prints name
        print(self.name)

    def is_trainable(self):
        # Returns whether this is a trainable layer
        return self.trainable
