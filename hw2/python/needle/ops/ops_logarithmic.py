from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        z_max = array_api.max(Z,axis=1,keepdims=True)
        logsumexp_z =  array_api.log(array_api.sum(array_api.exp(Z - z_max),axis=1,keepdims=True)) + z_max
        return Z - logsumexp_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        maxz = Tensor(array_api.max(z.realize_cached_data(),axis=1,keepdims=True))
        expz = exp(z - maxz)
        sumexpz = summation(expz,axes=(1,)).reshape((z.shape[0],1))
        # 对z求导是单位矩阵，不是1
        return out_grad - summation(out_grad,axes=(1,)).reshape((z.shape[0],1)) * expz / sumexpz
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z:numpy.ndarray):
        ### BEGIN YOUR SOLUTION
        z_max = array_api.max(Z,axis=self.axes,keepdims=True)
        return array_api.log(array_api.sum(array_api.exp(Z - z_max),axis=self.axes)) + \
                array_api.squeeze(z_max,axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        maxz = Tensor(array_api.max(z.realize_cached_data(),axis=self.axes,keepdims=True))
        expz = exp(z - maxz)
        sumexpz = summation(expz,axes=self.axes)
        # since these fucking functions don't have `keepdim` parameters, I should do it manually.
        z_shape = list(z.shape)
        if self.axes:
            for keepdim in self.axes:
                z_shape[keepdim] = 1
            sumexpz = reshape(sumexpz,shape=tuple(z_shape))
            # error broadcasting when doing multiplication, should reshape
            out_grad = reshape(out_grad,shape=tuple(z_shape))
        return out_grad * (expz / sumexpz)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

