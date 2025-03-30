import sys

sys.path.append("./python")
sys.path.append("./apps")
from simple_ml import *
import numdifftools as nd

import numpy as np
import mugrade
import needle as ndl

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [
            x.numpy()
            for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)
        ]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i]) for i in range(len(args))
    )
    return computed_grads,numerical_grads


# print(a)
# a = np.random.randn(5,4)
# b = np.random.randn(6,6,4,3)
# c_grad,n_grad = gradient_check(
#         ndl.matmul,
#         ndl.Tensor(a),
#         ndl.Tensor(b),
#     )
# print(c_grad[0].shape)
# print(n_grad[0].shape)


# a = np.sum(a,axis=(1,),keepdims=True)
# print(a.shape)
# print(np.broadcast_to(a,shape=(5,2)))

a = [].append(2)
print(a)