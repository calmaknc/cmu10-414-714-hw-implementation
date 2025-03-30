"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
import math


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=self.in_features,
                                                     fan_out=self.out_features,
                                                     device=device,
                                                     dtype=dtype))
        self.bias = None if bias is False else Parameter(ops.transpose(init.kaiming_uniform(
            fan_in=self.out_features,
            fan_out=1,
            device=device,
            dtype=dtype
        )))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X.matmul(self.weight)
        if self.bias:
            return out + self.bias.broadcast_to(out.shape)
        else:
            return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        shape = math.prod(X.shape)
        return ops.reshape(X,shape=(X.shape[0],shape // X.shape[0]))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        res = x
        for module in self.modules:
            res = module(res)
        return res
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch,class_num = logits.shape
        total_loss = ops.logsumexp(logits,axes=(1,)) - ops.summation(logits * init.one_hot(class_num,y,device=logits.device),axes=(1,))
        return ops.summation(total_loss) / batch
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim,device=device,dtype=dtype))
        self.bias = Parameter(init.zeros(dim,device=device,dtype=dtype))
        self.running_mean = init.zeros(dim, device=device)
        self.running_var = init.ones(dim, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # input: N * L  assert L == self.dim
        if self.training:
            batch_mean = x.sum(axes = (0,)) / x.shape[0] # (num_features, )
            batch_mean_vec = batch_mean.reshape((1, x.shape[1])) # (1, num_features)
            batch_var = (((x - batch_mean_vec.broadcast_to(x.shape))**2).sum((0,)) / x.shape[0]) # (num_features, )
            batch_var_vec = batch_var.reshape((1, x.shape[1])) # (1, num_features)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            norm = (x - batch_mean_vec.broadcast_to(x.shape)) / (batch_var_vec.broadcast_to(x.shape) + self.eps)**0.5
            return self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape) * norm \
                 + self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)
        else:
            norm = (x - self.running_mean.reshape((1, x.shape[1])).broadcast_to(x.shape)) / (self.running_var.reshape((1, x.shape[1])).broadcast_to(x.shape) + self.eps)**0.5
            return self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape) * norm + self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim,requires_grad=True,device=device,dtype=dtype))
        self.bias = Parameter(init.zeros(dim,requires_grad=True,device=device,dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = (x.sum(axes=(1,)).reshape((x.shape[0],1)) / x.shape[1]).broadcast_to(x.shape)
        var = (ops.summation((x - mean) ** 2,axes=(1,)).reshape((x.shape[0],1)) / x.shape[1]).broadcast_to(x.shape)
        return self.weight.reshape((1,self.dim)).broadcast_to(x.shape) * ((x - mean) / (var + self.eps) ** 0.5) + self.bias.reshape((1,self.dim)).broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            return x * init.randb(*x.shape, p=1-self.p, dtype=x.dtype,device=x.device) / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
