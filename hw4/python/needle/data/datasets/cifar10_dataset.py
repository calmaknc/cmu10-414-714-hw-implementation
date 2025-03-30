import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        
        if train:
            self.X = []
            self.y = []
            data_paths = [f"data_batch_{i}" for i in range(1,6,1)]
            for data_path in data_paths:
                with open(os.path.join(base_folder,data_path),'rb') as f:
                    data_dict = pickle.load(f,encoding='bytes')
                    # print(data_dict)
                    self.X.append(data_dict[b'data']) # list of numpy arrays
                    self.y.append(data_dict[b'labels'])

            self.X = np.concatenate(self.X,axis=0)
            self.y = np.concatenate(self.y,axis=0)
        else:
            with open(os.path.join(base_folder,"test_batch"),'rb') as f:
                data_dict = pickle.load(f,encoding='bytes')
                self.X = data_dict[b'data']
                self.y = data_dict[b'labels']
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        image,label = self.X[index].reshape((3,32,32)) / 255. ,self.y[index]
        if self.transforms:
            for transform in self.transforms:
                image = transform(image)
        return image,label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
