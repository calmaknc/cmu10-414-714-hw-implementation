"""Optimization module"""
import needle as ndl
import numpy as np
from collections import defaultdict

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = defaultdict(float)
        self.weight_decay = weight_decay

    def step(self):
        # ### BEGIN YOUR SOLUTION
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data    
            grad_momentum = self.momentum * self.u[param] + (1-self.momentum) * grad 
            # momentum should take weight decay into consideration
            self.u[param] = grad_momentum
            param.data -= self.lr * grad_momentum
        # ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = defaultdict(float)
        self.v = defaultdict(float)

    def step(self):
        ### BEGIN YOUR SOLUTION        
        self.t += 1
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (grad ** 2)
            m_correct = self.m[param] / (1 - self.beta1 ** self.t) 
            v_correct = self.v[param] / (1 - self.beta2 ** self.t)
            param.data = param.data - self.lr * m_correct / (v_correct ** 0.5 + self.eps) 
        ### END YOUR SOLUTION
