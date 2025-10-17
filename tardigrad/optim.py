"""
Optimization algorithms for training neural networks.

Implements stochastic gradient descent (SGD) with configurable learning rate
and automatic gradient zeroing after parameter updates.
"""

class SGD:
    """Stochastic gradient descent optimizer."""

    def __init__(self, params, alpha=0.1):
        self.params = params
        self.alpha = alpha

    def zero(self):
        for p in self.params:
            p.grad *= 0.0
    
    def step(self, zero=True):
        for p in self.params:
            p.data -= p.grad * self.alpha
            if zero:
                p.grad *= 0
