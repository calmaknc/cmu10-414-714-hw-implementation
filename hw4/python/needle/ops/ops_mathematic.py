"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from itertools import zip_longest
# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a, b):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs = node.inputs
        return (out_grad * rhs * power(lhs,rhs - 1), out_grad * Log()(lhs)* node.cached_data)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs = node.inputs[0]
        return (out_grad * self.scalar * lhs ** self.scalar - 1,)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs = node.inputs
        return (out_grad / rhs, -out_grad * lhs / rhs / rhs)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (divide_scalar(out_grad,self.scalar),)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a:NDArray):
        ### BEGIN YOUR SOLUTION
        shape = list(range(len(a.shape)))
        if self.axes:
            i,j = self.axes
        else:
            i,j = len(a.shape) - 1, len(a.shape) - 2
        shape[i],shape[j] = shape[j],shape[i]
        return a.permute(tuple(shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_size = node.inputs[0].shape
        out_size = out_grad.shape
        ax = []
        for i in range(len(in_size)):
            if in_size[i] != out_size[i]:
                ax.append(i)
        return transpose(out_grad,ax)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        return reshape(out_grad,in_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.broadcast_to(self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ax = []
        in_shape = node.inputs[0].shape
        len2 = len(self.shape)
        for i,(in_s,out_s) in enumerate(zip_longest(reversed(in_shape),reversed(self.shape))):
            if in_s != out_s:
                ax.append(len2 - i - 1)
        return reshape(summation(out_grad,axes=tuple(ax)),shape=in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes,tuple):
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis)
            return a
        else:
            return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        broad_shape = list(in_shape)
        axes = range(len(in_shape)) if self.axes is None else self.axes
        if isinstance(axes,(range,tuple)):
            for i in axes:
                broad_shape[i] = 1
        else:
            broad_shape[axes] = 1
        out_grad = reshape(out_grad,broad_shape)
        return broadcast_to(out_grad,shape=in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs = node.inputs
        grad_lhs = matmul(out_grad,transpose(rhs))
        grad_rhs = matmul(transpose(lhs),out_grad)
        if grad_lhs.shape != lhs.shape:
            grad_lhs = summation(grad_lhs,axes=tuple(range(len(grad_lhs.shape)-2)))
        if grad_rhs.shape != rhs.shape:
            grad_rhs = summation(grad_rhs,axes=tuple(range(len(grad_rhs.shape)-2)))
        return (grad_lhs,grad_rhs)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * -1.0,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / node.inputs[0],)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * node.cached_data)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * (a >= 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node.cached_data
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 - node.cached_data ** 2)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        new_shape = list(args[0].shape)
        new_shape.insert(self.axis,len(args))
        slices = [slice(0,s) for s in new_shape]
        new_tensor = array_api.empty(shape=new_shape,device=args[0].device)
        for i,arg in enumerate(args):
            slices[self.axis] = slice(i,i+1,1)
            new_tensor[tuple(slices)] = arg
        return new_tensor
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad,axis=self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(0,s) for s in A.shape]
        tmp_slices = [slice(0,s) for s in A.shape]
        tmp_slices.pop(self.axis)
        tmp_slices = tuple(tmp_slices)
        out_tensors = []
        for i in range(A.shape[self.axis]):
            slices[self.axis] = slice(i,i+1,1)
            tmp = array_api.empty(shape=tuple(new_shape),device=A.device)
            tmp[tmp_slices] = A[tuple(slices)] # need squeeze
            out_tensors.append(tmp)
        # print(out_tensors[0].shape)
        return tuple(out_tensors)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad,self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dilated_shape = list(a.shape)
        slices = [slice(0,s) for s in a.shape]
        for axis in self.axes:
            dilated_shape[axis] += dilated_shape[axis] * self.dilation
            slices[axis] = slice(0,dilated_shape[axis],self.dilation+1)
        out = array_api.full(shape=tuple(dilated_shape),fill_value=0.0,device=a.device)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        undilated_shape = list(a.shape)
        slices = [slice(0,s,1) for s in a.shape]
        for axis in self.axes:
            undilated_shape[axis] = undilated_shape[axis] // (self.dilation + 1)
            slices[axis] = slice(0,a.shape[axis],self.dilation+1)
        out = array_api.empty(shape=tuple(undilated_shape),device=a.device)        
        out = a[tuple(slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,self.axes,self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A -> n*(h-k+1)*(w-k+1)*k*k*Cin
        # K -> (K*K*Cin) * Cout
        A = A.pad(((0,0),(self.padding,self.padding),(self.padding,self.padding),(0,0)))
        N,H,W,C_in = A.shape
        K,_,_,C_out = B.shape
        Ns,Hs,Ws,Cs = A.strides

        inner_dim = K * K * C_in
        H_out, W_out = (H - K) // self.stride + 1, (W - K) // self.stride + 1
        A = A.as_strided(shape=(N, H_out, W_out, K, K, C_in),
                        strides=(Ns, self.stride * Hs, self.stride * Ws, Hs, Ws, Cs)).compact().reshape((N*H_out*W_out,inner_dim))
        # it seems A,B,out has to be compact here, why?
        out = A @ B.compact().reshape((inner_dim,C_out))
        return out.compact().reshape((N,H_out,W_out,C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X,W = node.inputs
        K = W.shape[0]
        if self.stride > 1:
            out_grad = dilate(out_grad,axes=(1,2),dilation = self.stride - 1)
        Xgrad = conv(out_grad,transpose(flip(W,axes=(0,1)),axes=(2,3)),padding=K-1-self.padding)
        Wgrad = conv(transpose(X,axes=(0,3)),transpose(transpose(out_grad,axes=(0,1)),axes=(1,2)),padding=self.padding)
        Wgrad = transpose(transpose(Wgrad,(0,1)),(1,2))
        return Xgrad,Wgrad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
