from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset
import os
import pickle
import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class MyDataset(Dataset):
    def __init__(self, root, train=True):
        self.root = root
        if train:
            # data_files = [os.path.join(root, "data_batch_{}".format(i)) for i in range(1, 6)]
            data_file = []
            for i in range(1,6):
                path = os.path.join(root, "data_batch_{}".format(i))
                data_file.append(path)
        else:
            data_file = [os.path.join(root, "test_batch")]
        self.images = []
        self.labels = []
        for path in data_file:
            with open(path, "rb") as fo:
                data = pickle.load(fo, encoding="bytes")
                self.images.extend(data[b"data"])
                self.labels.extend(data[b"labels"])
        # print(len(self.images), len(self.labels))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        image = self.images[idx].reshape(3,32,32) # reshape về 3 chiều
        image = np.transpose(image, (1, 2, 0))  # chuyển về(32,32,3)
        image = ToTensor()(image)
        label = self.labels[idx]
        return image, label

if __name__ =="__main__":
    dataset_train = MyDataset(root="./data/cifar-10-batches-py", train=True)
    train_loader = DataLoader(
        dataset = dataset_train,
        batch_size = 16,
        shuffle = True,
        drop_last = False,
    )
    for images, labels in train_loader:
        print(images.shape)
        print(labels)