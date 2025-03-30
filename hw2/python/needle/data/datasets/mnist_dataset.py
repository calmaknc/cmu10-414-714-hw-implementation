from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        with gzip.open(image_filename,'rb') as f_image:
            _,num_image,row,col = struct.unpack('>iiii',f_image.read(16))
            self.images = np.frombuffer(f_image.read(),dtype=np.uint8).reshape(num_image,row*col).astype(np.float32) / 255.0

        with gzip.open(label_filename,'rb') as f_label:
            _,_ = struct.unpack('>ii',f_label.read(8))
            self.labels = np.frombuffer(f_label.read(),dtype=np.uint8)

        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        image,label = self.images[index],self.labels[index]
        if self.transforms:
            # fucking reshape!!! fucking transform!!!!
            if isinstance(index,int):
                image = self.apply_transforms(image.reshape(28,28,1))
                image = image.reshape((784,))
            else:
                image = self.apply_transforms(image.reshape(28,28,-1))
                for i in range(image.shape[-1]):
                    image[i] = image[i].reshape((784,))
        return (image,label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.labels)
        ### END YOUR SOLUTION