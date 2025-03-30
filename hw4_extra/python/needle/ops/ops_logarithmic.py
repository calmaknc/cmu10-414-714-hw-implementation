from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        z_max = Z.max(axis=1,keepdims=True)
        self.maxz = z_max
        logsumexp_z =  array_api.log(array_api.sum(array_api.exp(Z - z_max),axis=1,keepdims=True)) + z_max
        return Z - logsumexp_z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        maxz = self.maxz
        expz = exp(z - maxz)
        sumexpz = array_api.sum(expz,axis=1,keepdims=True)
        return out_grad - out_grad.sum(axis=1,keepdims=True) * expz / sumexpz
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        z_max = Z.max(axis=self.axes,keepdims=True)
        self.maxz = z_max
        zmax_c = Z.max(axis=self.axes,keepdims=False)
        return array_api.log(array_api.exp(Z - z_max.broadcast_to(Z.shape)).sum(axis=self.axes,keepdims=False)) + zmax_c
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # z = node.inputs[0]
        # maxz = self.maxz
        # expz = exp(z - maxz.broadcast_to(z.shape))
        # sumexpz = expz.sum(axes=self.axes)
        # # since these fucking functions don't have `keepdim` parameters, I should do it manually.
        # z_shape = list(z.shape)
        # if self.axes:
        #     for keepdim in self.axes:
        #         z_shape[keepdim] = 1
        #     sumexpz = reshape(sumexpz,shape=tuple(z_shape))
        #     # error broadcasting when doing multiplication, should reshape
        #     out_grad = reshape(out_grad,shape=tuple(z_shape))
        # return out_grad * (expz / sumexpz)
        z = node.inputs[0]
        max_z = Tensor(z.realize_cached_data().max(axis=self.axes, keepdims=True), device=z.device)
        exp_z = exp(z - max_z.broadcast_to(z.shape))
        sum_exp_z = summation(exp_z, axes=self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

