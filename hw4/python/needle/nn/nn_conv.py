"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=21,
                                                     fan_out=21,
                                                     shape=(self.kernel_size,self.kernel_size,in_channels,out_channels),
                                                     dtype=dtype,
                                                     device=device))
        if bias:
            interval = 1.0 / math.sqrt(in_channels* kernel_size**2)
            self.bias = Parameter(init.rand(out_channels,
                                            low=-interval,
                                            high=interval,
                                            dtype=dtype,
                                            device=device))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # x: NCHW implemented:NHWC
        x = x.transpose(axes=(1,2)).transpose(axes=(2,3))
        # (H + 2*P -K) // S + 1
        out = ops.conv(x,self.weight,stride=self.stride,padding=(self.kernel_size - 1) // 2)
        if self.bias is not None:
            out = out + self.bias.reshape((1,1,1,self.out_channels)).broadcast_to(out.shape) 
        return out.transpose(axes=(2,3)).transpose(axes=(1,2))
        ### END YOUR SOLUTION