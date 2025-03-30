import numpy as np
import numdifftools as nd
import gzip
import struct
from numpy.linalg import norm
def parse_mnist(image_filename, label_filename):
    ### BEGIN YOUR CODE
    # struct module performs conversions between Python values and C structs represented as Python bytes objects.  
    # parse image
    with gzip.open(image_filename,'rb') as f_image:
        _,num_image,row,col = struct.unpack('>iiii',f_image.read(16))
        images = np.frombuffer(f_image.read(),dtype=np.uint8).reshape(num_image,row*col).astype(np.float32) / 255.0

    #parse label
    with gzip.open(label_filename,'rb') as f_label:
        _,_ = struct.unpack('>ii',f_label.read(8))
        labels = np.frombuffer(f_label.read(),dtype=np.uint8)

    return (images,labels)
    ### END YOUR CODE


X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                    "data/train-labels-idx1-ubyte.gz")
theta = np.zeros((X.shape[1], y.max()+1), dtype=np.float32)


def softmax_regression_epoch(X:np.ndarray[np.float32], 
                             y:np.ndarray[np.uint8], 
                             theta:np.ndarray[np.float32], 
                             lr = 0.1, 
                             batch=100):
    for i in range(0,len(y),batch):
        X_batch = X[i:i+batch]
        y_batch = y[i:i+batch]
        Z = np.matmul(X_batch,theta)
        print(norm(Z))
        Z = np.exp(Z)
        print(norm(Z))
        Z = Z / np.sum(Z,axis=1,keepdims=True)
        print(norm(Z))
        I_y = np.zeros(shape=(batch,theta.shape[1]),dtype=np.uint8)
        I_y[np.arange(batch),y_batch] = 1
        print(norm(Z - I_y))
        grad = (np.matmul(X_batch.transpose(),(Z - I_y)))  / batch
        print(norm(grad))
        theta -= lr * grad
        print(norm(theta))
        print('\n')
    ### END YOUR CODE

softmax_regression_epoch(X[:100], y[:100], theta, lr=0.1, batch=10)