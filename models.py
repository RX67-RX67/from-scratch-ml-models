import numpy as np

def softmax(x):
    if x.ndim == 1:  
        exp_x = np.exp(x - np.max(x))  
        return exp_x / np.sum(exp_x)
    elif x.ndim == 2:  
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    else:
        raise ValueError("Input must be a 1D or 2D numpy array")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def activation_function(x, method:str = "ReLU"):
    if method == "ReLU":
        return np.maximum(0, x)
    elif method == "Sigmoid":
        return sigmoid(x)
    else:
        raise ValueError("Unsupported activation function. Please choose either ReLU or Sigmoid")

class Model:

    def __init__(self, input_dim, output_dim, architecture = None, activation = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.parameters = {}

        self.architecture = architecture
        self.activation = activation
    
    def get_parameters(self):
        return self.parameters
    
    def flatten_parameters(self):
        flat = []
        for v in self.parameters.values():
            flat.extend(v.flatten())
        return np.array(flat)
    
    def update_parameters(self, flat_array):
        raise NotImplementedError('all subclasses must implement update_parameters() method')
    
    def forward(self, x):
        raise NotImplementedError('all subclasses must implement forward() method')

class LinearRegression(Model):

    def __init__(self, input_dim, output_dim, architecture=None, activation=None):
        super().__init__(input_dim, output_dim, architecture, activation)
        self.parameters['w'] = np.random.randn(self.input_dim, self.output_dim) * 0.01
        self.parameters['b'] = np.zeros((self.output_dim,))
    
    def forward(self, x):
        w = self.parameters['w']
        b = self.parameters['b']
        return np.dot(x, w) + b
    
    def update_parameters(self, flat_array):
        w_size = self.output_dim * self.input_dim
        self.parameters['w'] = flat_array[:w_size].reshape(self.input_dim, self.output_dim)
        self.parameters['b'] = flat_array[w_size:].reshape(self.output_dim,)

class LogisticRegression(Model):

    def __init__(self, input_dim, output_dim, architecture=None, activation=None):
        super().__init__(input_dim, output_dim, architecture, activation)
        self.parameters['w'] = np.random.randn(self.input_dim, self.output_dim) * 0.01
        self.parameters['b'] = np.zeros((self.output_dim,))
    
    def forward(self, x):
        w = self.parameters['w']
        b = self.parameters['b']
        z = np.dot(x, w) + b
        return sigmoid(z)
    
    def update_parameters(self, flat_array):
        w_size = self.output_dim * self.input_dim
        self.parameters['w'] = flat_array[:w_size].reshape(self.input_dim, self.output_dim)
        self.parameters['b'] = flat_array[w_size:].reshape(self.output_dim,)

class DenseFeedForwardNetwork(Model):

    def __init__(self, input_dim, output_dim, architecture=None, activation=None):
        super().__init__(input_dim, output_dim, architecture, activation)
        self.architecture = architecture 
        self.activation = activation
        
        layer_dim = [input_dim] + self.architecture + [output_dim]
        for i in range(1, len(layer_dim)):
            self.parameters[f'w{i}'] = np.random.randn(layer_dim[i-1], layer_dim[i]) * 0.01
            self.parameters[f'b{i}'] = np.zeros((layer_dim[i],))
        
    def forward(self, x):
        layer_dim = [self.input_dim] + self.architecture + [self.output_dim]

        a = x
        layer_total = len(layer_dim) - 1

        for i in range(1, layer_total+1):

            w = self.parameters[f'w{i}']
            b = self.parameters[f'b{i}']
            z = np.dot(a, w) + b

            if i == layer_total:
                a = z
            else:
                if self.activation == "ReLU":
                    a = activation_function(z, method='ReLU')
                elif self.activation == "Sigmoid":
                    a = activation_function(z, method="Sigmoid")
                else:
                    raise ValueError(f"Unsupported activation function{self.activation}")

        return softmax(a)

    def update_parameters(self, flat_array):
        layer_dim = [self.input_dim] + self.architecture + [self.output_dim]
        start = 0

        for i in range(1, len(layer_dim)):

            w_size = layer_dim[i] * layer_dim[i-1]
            b_size = layer_dim[i]

            self.parameters[f'w{i}'] = flat_array[start:start+w_size].reshape(layer_dim[i-1], layer_dim[i])
            start += w_size

            self.parameters[f'b{i}'] = flat_array[start:start+b_size].reshape(layer_dim[i],)
            start += b_size
        
