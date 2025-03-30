#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

void matmul(const float *X, size_t rows_x, size_t cols_x, const float *Y, size_t cols_y, float *res);
void normalize_row(float *X, size_t rows, size_t cols);
void matmul_transpose(const float *X, size_t rows_x, size_t cols_x, const float *Y, size_t cols_y, float *res);

void print_pointer(const float *x, size_t size)
{
    for (size_t i = 0; i < size; ++i)
    {
        std::cout << x[i] << ' ';
    }
    std::cout << '\n';
    std::cout << '\n';
}

void norm(const float *x, size_t size)
{
    double n = 0.0;
    for (size_t i = 0; i < size; ++i)
    {
        n += x[i] * x[i];
    }
    std::cout << sqrt(n) << '\n';
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
                                  float *theta, size_t m, size_t n, size_t k,
                                  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // Z size batch*k
    float *Z = new float[batch * k]();
    // grad size (n,k)
    float *grad = new float[n * k]();
    for (size_t i = 0; i < m; i += batch)
    {
        // calculate z
        matmul(X + i * n, batch, n, theta, k, Z);
        // norm(Z, batch * k);
        for (size_t a = 0; a < batch * k; a++)
        {
            Z[a] = exp(Z[a]);
        }
        // norm(Z, batch * k);
        //  normalize z
        normalize_row(Z, batch, k);
        // norm(Z, batch * k);
        //  calculate z - I_y
        for (size_t j = 0; j < batch; j++)
        {
            Z[j * k + (y + i)[j]] -= 1.0;
        }
        // norm(Z, batch * k);
        //  calculate grad
        matmul_transpose(X + i * n, batch, n, Z, k, grad);
        // norm(grad, n * k);
        for (size_t j = 0; j < n * k; j++)
        {
            theta[j] -= (lr * grad[j] / batch);
        }
        // norm(theta, n * k);
        // std::cout << '\n';
    }
    delete[] Z;
    delete[] grad;
    /// END YOUR CODE
}

void matmul(const float *X, size_t rows_x, size_t cols_x, const float *Y, size_t cols_y, float *res)
{
    for (size_t row = 0; row < rows_x; ++row)
    {
        for (size_t col = 0; col < cols_y; ++col)
        {
            res[row * cols_y + col] = 0.0;
            for (size_t a = 0; a < cols_x; ++a)
            {
                res[row * cols_y + col] += X[row * cols_x + a] * Y[a * cols_y + col];
            }
        }
    }
}

void matmul_transpose(const float *X, size_t rows_x, size_t cols_x, const float *Y, size_t cols_y, float *res)
{
    for (size_t row = 0; row < cols_x; ++row)
    {
        for (size_t col = 0; col < cols_y; ++col)
        {
            res[row * cols_y + col] = 0.0;
            for (size_t a = 0; a < rows_x; ++a)
            {
                res[row * cols_y + col] += X[a * cols_x + row] * Y[a * cols_y + col];
            }
        }
    }
}

void normalize_row(float *X, size_t rows, size_t cols)
{
    for (size_t row = 0; row < rows; row++)
    {
        float sum = 0;
        for (size_t col = 0; col < cols; col++)
        {
            sum += X[row * cols + col];
        }
        for (size_t col = 0; col < cols; col++)
        {
            X[row * cols + col] /= sum;
        }
    }
}

/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m)
{
    m.def("softmax_regression_epoch_cpp", [](py::array_t<float, py::array::c_style> X, py::array_t<unsigned char, py::array::c_style> y, py::array_t<float, py::array::c_style> theta, float lr, int batch)
          { softmax_regression_epoch_cpp(
                static_cast<const float *>(X.request().ptr),
                static_cast<const unsigned char *>(y.request().ptr),
                static_cast<float *>(theta.request().ptr),
                X.request().shape[0],
                X.request().shape[1],
                theta.request().shape[1],
                lr,
                batch); }, py::arg("X"), py::arg("y"), py::arg("theta"), py::arg("lr"), py::arg("batch"));
}
