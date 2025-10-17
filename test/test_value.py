"""
Test suite for Value class operations and gradients
"""

import pytest
import math
from tardigrad.value import Value


class TestValueOperations:
    """Test forward pass operations for Value"""

    def test_addition(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        assert c.data == 5.0

    def test_addition_with_scalar(self):
        a = Value(2.0)
        c = a + 3
        assert c.data == 5.0

    def test_radd(self):
        a = Value(2.0)
        c = 3 + a
        assert c.data == 5.0

    def test_multiplication(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        assert c.data == 6.0

    def test_multiplication_with_scalar(self):
        a = Value(2.0)
        c = a * 3
        assert c.data == 6.0

    def test_rmul(self):
        a = Value(2.0)
        c = 3 * a
        assert c.data == 6.0

    def test_power(self):
        a = Value(2.0)
        c = a ** 3
        assert c.data == 8.0

    def test_power_float(self):
        a = Value(4.0)
        c = a ** 0.5
        assert abs(c.data - 2.0) < 1e-6

    def test_negation(self):
        a = Value(2.0)
        c = -a
        assert c.data == -2.0

    def test_subtraction(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        assert c.data == 2.0

    def test_subtraction_with_scalar(self):
        a = Value(5.0)
        c = a - 3
        assert c.data == 2.0

    def test_rsub(self):
        a = Value(3.0)
        c = 5 - a
        assert c.data == 2.0

    def test_division(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        assert abs(c.data - 2.0) < 1e-6

    def test_division_with_scalar(self):
        a = Value(6.0)
        c = a / 3
        assert abs(c.data - 2.0) < 1e-6

    def test_rdiv(self):
        a = Value(3.0)
        c = 6 / a
        assert abs(c.data - 2.0) < 1e-6

    def test_relu_positive(self):
        a = Value(2.0)
        c = a.relu()
        assert c.data == 2.0

    def test_relu_negative(self):
        a = Value(-2.0)
        c = a.relu()
        assert c.data == 0.0

    def test_relu_zero(self):
        a = Value(0.0)
        c = a.relu()
        assert c.data == 0.0

    def test_sigmoid(self):
        a = Value(0.0)
        c = a.sigmoid()
        expected = 1 / (1 + math.exp(0))
        assert abs(c.data - expected) < 1e-6

    def test_tanh(self):
        a = Value(0.0)
        c = a.tanh()
        assert abs(c.data - 0.0) < 1e-6

    def test_tanh_positive(self):
        a = Value(1.0)
        c = a.tanh()
        expected = (math.exp(2.0) - 1) / (math.exp(2.0) + 1)
        assert abs(c.data - expected) < 1e-6

    def test_gelu(self):
        a = Value(0.0)
        c = a.gelu()
        assert abs(c.data - 0.0) < 1e-6

    def test_complex_expression(self):
        a = Value(2.0)
        b = Value(3.0)
        c = Value(-1.0)
        d = a * b + c
        assert d.data == 5.0


class TestValueGradients:
    """Test backward pass (gradients) for Value"""

    def test_addition_gradient(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a + b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_multiplication_gradient(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        c.backward()
        assert a.grad == 3.0
        assert b.grad == 2.0

    def test_power_gradient(self):
        a = Value(2.0)
        c = a ** 3
        c.backward()
        # d/da (a^3) = 3*a^2 = 3*4 = 12
        assert abs(a.grad - 12.0) < 1e-6

    def test_negation_gradient(self):
        a = Value(2.0)
        c = -a
        c.backward()
        assert a.grad == -1.0

    def test_subtraction_gradient(self):
        a = Value(5.0)
        b = Value(3.0)
        c = a - b
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_division_gradient(self):
        a = Value(6.0)
        b = Value(3.0)
        c = a / b
        c.backward()
        # dc/da = 1/b = 1/3
        assert abs(a.grad - 1.0/3.0) < 1e-6
        # dc/db = -a/b^2 = -6/9 = -2/3
        assert abs(b.grad - (-2.0/3.0)) < 1e-6

    def test_relu_gradient_positive(self):
        a = Value(2.0)
        c = a.relu()
        c.backward()
        assert a.grad == 1.0

    def test_relu_gradient_negative(self):
        a = Value(-2.0)
        c = a.relu()
        c.backward()
        assert a.grad == 0.0

    def test_relu_gradient_zero(self):
        a = Value(0.0)
        c = a.relu()
        c.backward()
        assert a.grad == 0.0

    def test_sigmoid_gradient(self):
        a = Value(0.0)
        c = a.sigmoid()
        c.backward()
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        s = 1 / (1 + math.exp(0))
        expected_grad = s * (1 - s)
        assert abs(a.grad - expected_grad) < 1e-6

    def test_tanh_gradient(self):
        a = Value(0.0)
        c = a.tanh()
        c.backward()
        # tanh'(0) = 1 - tanh(0)^2 = 1 - 0 = 1
        assert abs(a.grad - 1.0) < 1e-6

    def test_tanh_gradient_nonzero(self):
        a = Value(1.0)
        c = a.tanh()
        c.backward()
        t = (math.exp(2.0) - 1) / (math.exp(2.0) + 1)
        expected_grad = 1 - t**2
        assert abs(a.grad - expected_grad) < 1e-6

    def test_gelu_gradient(self):
        a = Value(0.0)
        c = a.gelu()
        c.backward()
        # gelu'(0) = cdf(0) + 0 * pdf(0) = 0.5
        assert abs(a.grad - 0.5) < 1e-6

    def test_chain_rule(self):
        a = Value(2.0)
        b = Value(3.0)
        c = a * b  # c = 6
        d = c + a  # d = 8
        d.backward()
        # dd/da = dd/dc * dc/da + dd/da = 1 * 3 + 1 = 4
        assert abs(a.grad - 4.0) < 1e-6
        # dd/db = dd/dc * dc/db = 1 * 2 = 2
        assert abs(b.grad - 2.0) < 1e-6

    def test_multi_use_gradient(self):
        a = Value(2.0)
        b = a + a  # b = 4
        b.backward()
        # db/da = 1 + 1 = 2 (a is used twice)
        assert abs(a.grad - 2.0) < 1e-6

    def test_complex_gradient(self):
        x = Value(-4.0)
        z = 2 * x + 2 + x
        q = z.relu() + z * x
        h = (z * z).relu()
        y = h + q + q * x
        y.backward()
        # This tests a complex computational graph
        # Just verify backward runs without error and grad is computed
        assert x.grad != 0.0

    def test_neuron_gradient(self):
        # Simulate a simple neuron: y = (w*x + b).tanh()
        x = Value(1.0)
        w = Value(0.5)
        b = Value(0.3)
        y = (w * x + b).tanh()
        y.backward()
        
        # All gradients should be non-zero
        assert x.grad != 0.0
        assert w.grad != 0.0
        assert b.grad != 0.0

    def test_gradient_accumulation(self):
        """Test that gradients accumulate properly"""
        a = Value(2.0)
        b = Value(3.0)
        
        # Use a in multiple operations
        c = a + b
        d = a * b
        e = c + d
        e.backward()
        
        # a.grad should be sum of gradients from both paths
        # de/da = de/dc * dc/da + de/dd * dd/da = 1*1 + 1*3 = 4
        assert abs(a.grad - 4.0) < 1e-6


class TestValueEdgeCases:
    """Test edge cases for Value"""

    def test_zero_multiplication(self):
        a = Value(0.0)
        b = Value(5.0)
        c = a * b
        c.backward()
        assert a.grad == 5.0
        assert b.grad == 0.0

    def test_power_zero(self):
        a = Value(5.0)
        c = a ** 0
        assert c.data == 1.0
        c.backward()
        assert a.grad == 0.0

    def test_power_one(self):
        a = Value(5.0)
        c = a ** 1
        assert c.data == 5.0
        c.backward()
        assert a.grad == 1.0

    def test_large_values(self):
        a = Value(1000.0)
        b = Value(2000.0)
        c = a + b
        assert c.data == 3000.0

    def test_small_values(self):
        a = Value(1e-10)
        b = Value(2e-10)
        c = a + b
        assert abs(c.data - 3e-10) < 1e-15

    def test_negative_power(self):
        a = Value(2.0)
        c = a ** -1
        assert abs(c.data - 0.5) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
