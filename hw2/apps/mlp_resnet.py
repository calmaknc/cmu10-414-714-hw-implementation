import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    resblock_list = [
        nn.Linear(in_features=dim,out_features=hidden_dim),
        norm(dim=hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(in_features=hidden_dim,out_features=dim),
        norm(dim=dim),
    ]
    
    return nn.Sequential(nn.Residual(nn.Sequential(*resblock_list)),nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    mlpresnet_list = [
        nn.Linear(dim,hidden_dim),
        nn.ReLU(),
    ]
    for _ in range(num_blocks):
        block = ResidualBlock(hidden_dim,hidden_dim // 2,norm,drop_prob)
        mlpresnet_list.append(block)
    mlpresnet_list.append(nn.Linear(hidden_dim,num_classes))
    return nn.Sequential(*mlpresnet_list)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # return avg error rate(float) and avg loss(float)
    total_loss, total_error = [],0.0
    loss_fn = nn.SoftmaxLoss()
    num = len(dataloader.dataset)
    if opt is None:
        model.eval()    
        for images,labels in dataloader:
            predict_logits = model(images)
            loss = loss_fn(predict_logits,labels)
            total_error += np.sum(predict_logits.numpy().argmax(axis=1) != labels.numpy())
            total_loss.append(loss.numpy())
    else:
        model.train()
        for images,labels in dataloader:
            predict_logits = model(images)
            loss = loss_fn(predict_logits,labels)
            total_error += np.sum(predict_logits.numpy().argmax(axis = 1) != labels.numpy())
            total_loss.append(loss.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    
    return total_error/num,np.mean(total_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    train_set = ndl.data.MNISTDataset(image_filename=data_dir+"/train-images-idx3-ubyte.gz",
                                    label_filename=data_dir+"/train-labels-idx1-ubyte.gz")
    test_set = ndl.data.MNISTDataset(image_filename=data_dir+"/t10k-images-idx3-ubyte.gz",
                                    label_filename=data_dir+"/t10k-labels-idx1-ubyte.gz")
    train_loader,test_loader = ndl.data.DataLoader(train_set,batch_size=batch_size,shuffle=True),ndl.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)
    net = MLPResNet(784,hidden_dim=hidden_dim)
    opt = optimizer(net.parameters(),lr=lr,weight_decay=weight_decay)
    for _ in range(epochs):
        train_error,train_loss = epoch(dataloader=train_loader,model=net,opt=opt)
    test_error,test_loss = epoch(dataloader=test_loader,model=net,opt=None)
    return train_error,train_loss,test_error,test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
