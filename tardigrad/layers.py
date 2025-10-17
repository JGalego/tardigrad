"""
Neural network layer implementations.

Provides foundational building blocks for constructing neural networks:
- Layer: Base class for all layers
- Linear: Fully connected layer with weights and biases
- Sequential: Container for chaining multiple layers together
"""

import numpy as np

from tardigrad.tensor import Tensor

class Layer:
    """Base class for neural network layers."""

    def __init__(self):
        self.params = []
    
    def get_params(self):
        return self.params


class Linear(Layer):
    """Fully connected linear layer."""

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.rand(n_inputs, n_outputs)
        self.weights = Tensor(W)
        self.biases = Tensor(np.zeros(n_outputs))
        self.params.append(self.weights)
        self.params.append(self.biases)
    
    def forward(self, input):
        return input.matmul(self.weights) + self.biases.expand(0, len(input.data))


class Sequential(Layer):
    """Container for sequentially applying multiple layers."""

    def __init__(self, layers=[]):
        super().__init__()
        self.layers = layers
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output

    def get_params(self):
        params = []
        for layer in self.layers:
            params += layer.get_params()
        return params
