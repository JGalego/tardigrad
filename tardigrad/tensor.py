"""
Multi-dimensional tensor autograd engine.

Implements automatic differentiation for NumPy-backed tensors with support for
matrix operations (matmul, transpose, sum), broadcasting, activation functions
(tanh, ReLU, sigmoid, GELU), and backpropagation through computational graphs.

Adapted from Andrew Trask's Grokking Deep Learning:
> Chapter 13 - Introducing Automating Optimization
"""

import numpy as np

from scipy.special import erf

class Tensor:
    """Multi-dimensional array with automatic differentiation support."""

    def __init__(self, data, prev=(), op="", label=""):
        self.data = np.array(data, dtype=np.float64)
        self.prev = prev
        self.op = op
        self.label = label
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
    
    def __add__(self, other):
        output = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output
    
    def __mul__(self, other):
        output = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad

        output._backward = _backward
        return output
    
    def __pow__(self, other):
        output = Tensor(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other - 1)) * output.grad
        
        output._backward = _backward
        return output
    
    def __neg__(self):
        return self * Tensor(-np.ones_like(self.data))
    
    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1
    
    def relu(self):
        output = Tensor((self.data > 0) * self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * output.grad
        
        output._backward = _backward
        return output
    
    def sigmoid(self):
        x = self.data
        s = 1 / (1 + np.exp(-x))
        output = Tensor(s, (self,), 'Sigmoid')

        def _backward():
            self.grad += s * (1 - s) * output.grad
        
        output._backward = _backward
        return output
  
    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        output = Tensor(t, (self,), 'Tanh')

        def _backward():
            self.grad += (1 - t**2) * output.grad
        
        output._backward = _backward
        return output

    def gelu(self):
        x = self.data
        cdf = 0.5 * (1.0 + erf(x / np.sqrt(2.0)))
        pdf = np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
        g = x * cdf
        output = Tensor(g, (self,), 'GeLU')

        def _backward():
            self.grad += (cdf + x * pdf) * output.grad
        
        self._backward = _backward
        return output
    
    def expand(self, dim, copies):
        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies)\
                            .reshape(list(self.data.shape) + [copies])\
                            .transpose(trans_cmd)
        output = Tensor(new_data, (self,), 'Expand_' + str(dim))

        def _backward():
            self.grad += output.grad.sum(dim)

        output._backward = _backward
        return output
    
    def sum(self, dim):
        output = Tensor(self.data.sum(dim), (self,), 'Sum_' + str(dim))

        def _backward():
            self.grad += Tensor(output.grad).expand(dim, self.data.shape[dim]).data
        
        output._backward = _backward
        return output
    
    def transpose(self):
        output = Tensor(self.data.transpose(), (self,), 'Transpose')

        def _backward():
            self.grad += output.grad.transpose()
        
        output._backward = _backward
        return output
    
    def matmul(self, x):
        output = Tensor(self.data.dot(x.data), (self, x), 'MatMul')

        def _backward():
            self.grad += output.grad.dot(x.data.transpose())
            x.grad += self.data.transpose().dot(output.grad)

        output._backward = _backward
        return output
    
    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
    
    def backward(self):
        topo = []
        seen = set()

        def topo_sort(root):
            if root not in seen:
                seen.add(root)
                for child in root.prev:
                    topo_sort(child)
                topo.append(root)
        
        topo_sort(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
