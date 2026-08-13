import numpy as np
import cupy as cp



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
        self.batch_size = batch_size
        self.rng = cp.random.default_rng(seed=42)
        self.weight = self.rng.normal(size=(in_features, out_features)) * cp.sqrt(1.0 / in_features)
        self.bias = self.rng.normal(size=(out_features,)) * cp.sqrt(1.0 / in_features)
        self.grad_weight = cp.zeros((in_features, out_features))
        self.grad_bias = cp.zeros(out_features)
        self.input = cp.zeros((batch_size, in_features))

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = input @ self.weight + self.bias
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



# ---------------------------------------------------------
# Linear layer with ADAM
# ---------------------------------------------------------
class LinearAdam:
    """A fully connected layer implemented with NumPy arrays."""

    def __init__(
        self, in_features: int, out_features: int, batch_size: int
    ) -> None:
        super(LinearAdam, self).__init__()
        self.batch_size = batch_size

        # For comparison: more primitive initialization of weights and bias
        # self.weight = cp.random.normal(size=(in_features, out_features)) * cp.sqrt(1.0 / in_features)
        # self.bias = cp.random.normal(size=(out_features,)) * cp.sqrt(1.0 / in_features)

        # Initialization of weights like in torch
        self.k = cp.sqrt(1/(in_features)) 
        self.rng = cp.random.default_rng() 
        self.weight = self.rng.uniform(-self.k,self.k, size=(in_features, out_features))
        self.bias = self.rng.uniform(-self.k,self.k, size=(out_features,))

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
        
        
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
        output = input @ self.weight + self.bias
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

# TODO Copied from the internet, not compatible yet
class Conv:

    def __init__(self, n_filters=1, filter_size=(3, 3), padding=0, stride=1, trainable=True):
        self.name = "conv"

        self.n_filters = n_filters #out channels
        self.padding = padding
        self.stride = stride
        self.filter_size = filter_size
        self.filters = None
        self.trainable = trainable


    def forward(self, img):
   
        self.channel_size = img.shape[2] if len(img.shape) == 3 else 1

        # Initialization
        if self.filters is None:    
            self.filters = cp.random.uniform(-1, 1, size=(self.n_filters, *self.filter_size, self.channel_size)) 


        # SET OUTPUT FILTER
        print(img.shape[0])
        out_height = ((img.shape[0] - self.filter_size[0] + 2 * self.padding) / self.stride) + 1
        out_width = ((img.shape[1] - self.filter_size[1] + 2 * self.padding) / self.stride) + 1
        
        
        # STRIDE CHECK
        if int(out_height) != out_height or int(out_width) != out_width:
            print(int(out_height),out_height,int(out_width),out_width)
            raise  Exception(f'Stride {self.stride} is incorrect')
            
            
        out = cp.zeros((int(out_height), int(out_width), self.n_filters))
        
    

        if self.padding:
            img = cp.pad(
                img,
                ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                mode="constant",
            )

        self.inp_last = img

        # CONV  
        for f in range(self.n_filters):
            for out_col, col in enumerate(range(0, img.shape[1] - self.filter_size[1] + 1, self.stride)):
                for out_row, row in enumerate(range(0, img.shape[0] - self.filter_size[0] + 1,self.stride)):
                
                    slic = img[row : self.filter_size[0] + row, 
                                col : self.filter_size[1] + col,]

                    out[out_row, out_col, f] = cp.sum(slic * self.filters[f])

            out += out
                
        return out

    def backward(self, dl_da):
        
        # RESHAPE gradiet as filter
        dl_da = dl_da.transpose(2, 0, 1)
        dl_da = dl_da[..., cp.newaxis]
        dl_da = cp.tile(dl_da, (1, 1, 1, self.channel_size))
        
        # SET output for previous layer and gradient of filter
        grad_a = cp.zeros((self.n_filters, *self.inp_last.shape))
        grad_f = cp.zeros_like(self.filters)
        
        


        # Update Filter
        for f in range(self.n_filters):
            for out_col, col in enumerate(range(0, self.inp_last.shape[1] - dl_da.shape[2] + 1, self.stride)):
                for out_row, row in enumerate(range(0, self.inp_last.shape[0] - dl_da.shape[1] + 1, self.stride)):
                
                    slic = self.inp_last[row : dl_da.shape[1] + row, 
                                          col : dl_da.shape[2] + col,]

                    grad_f[f, out_row, out_col] = cp.sum(slic * dl_da[f])
                
                
        # gradient for input      
        filte = self.filters
        padding = dl_da.shape[1] - 1
        pad_width = ((padding, padding), (padding, padding), (0, 0))
        
        for f in range(self.n_filters):
            
            padded_filter = cp.pad(cp.flip(filte[f], 0), pad_width, mode="constant", constant_values=0)
            
            for out_col, col in enumerate(range(0, padded_filter.shape[1] - dl_da.shape[2] + 1, self.stride)):
                for out_row, row in enumerate(range(0, padded_filter.shape[0] - dl_da.shape[1] + 1,self.stride)):
                
                    slic = padded_filter[row : dl_da.shape[1] + row, 
                                          col : dl_da.shape[2] + col,]

                    grad_a[f, out_row, out_col] = cp.sum(slic * dl_da[f])
                    
        grad_a = cp.mean(grad_a, axis=0)
    

        return grad_a if not self.trainable else (grad_a, (grad_f), (self.filters))

    def update(self, learning_rate) -> None:
        pass
    
    def get_weights(self):
        return [1]
    
    def set_weights(self, new_weights):
        pass
    


# TODO Copied from the internet, not compatible yet
# MAX Pooling layer 

class MaxPool:
    def __init__(self, filter_size=3, stride=1):
        self.filter_size = filter_size
        self.stride = stride
        self.trainable = None
    
    def forward(self, inp):
        
        self.inp_last = inp

        out_size = ((inp.shape[0] - self.filter_size) / self.stride) + 1
        
        if int(out_size) != out_size :
            raise  Exception(f'Stride {self.stride} is incorrect')
        else:
            out_size = int(out_size)
        
        out = np.zeros((out_size, out_size, inp.shape[2]))
        
        for channel in range(inp.shape[2]):
            for out_col, col in enumerate(range(0, inp.shape[1] - self.filter_size + 1, self.stride)):
                for out_row, row in enumerate(range(0, inp.shape[0] - self.filter_size + 1,self.stride)):
                
                    slic = inp[row : self.filter_size + row, 
                                col : self.filter_size + col,
                                channel]

                    out[out_row, out_col, channel] = np.max(slic)
                    
        
        return out
    
    def backward(self, dl_da):
        
        grad_a = np.zeros_like(self.inp_last)
        
        mask = self.inp_last
        

        for channel in range(self.inp_last.shape[2]):
            for out_col, col in enumerate(range(0, self.inp_last.shape[1] - dl_da.shape[1] + 1, self.stride)):
                for out_row, row in enumerate(range(0, self.inp_last.shape[0] - dl_da.shape[0] + 1,self.stride)):
                
                    slic = self.inp_last[row : dl_da.shape[0] + row, 
                                          col : dl_da.shape[1] + col,
                                          channel].copy()
                    
                    max_ind = np.unravel_index(np.argmax(slic), slic.shape)
                    slic = keep_ind_only(slic, max_ind)
                    
                    grad_a[row : dl_da.shape[0] + row, 
                           col : dl_da.shape[1] + col,
                           channel]                      += slic * dl_da[:, :, channel]
        return grad_a
    
    __call__ = forward



# ---------------------------------------------------------
# Sigmoid activation
# ---------------------------------------------------------
class Sigmoid:
    """Sigmoid activation function"""

    def __init__(self, batch_size: int) -> None:
        super(Sigmoid, self).__init__()
        self.input = cp.zeros(batch_size)
        self.output = cp.zeros(batch_size)

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        self.input = input
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

# ---------------------------------------------------------
# ReLU activation
# ---------------------------------------------------------
class Relu:
    """ReLU activation function"""

    def __init__(self, batch_size: int) -> None:
        super(Relu, self).__init__()
        self.input = cp.zeros(batch_size)
        self.output = cp.zeros(batch_size)

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
    
    
