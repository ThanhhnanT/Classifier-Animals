import cv2
from torch.utils.data import Dataset
import os
from torchvision.transforms import Resize, ToTensor, Compose
from torch.utils.data import DataLoader

class AnimalDataset(Dataset):

    def __init__(self, root, train=True, transform=None):
        self.transform = transform
        if train:
            mode = "train"
        else:
            mode = "test"
        self.root = os.path.join(root, mode)
        self.categories = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
        self.images_path = []
        self.labels = []
        for i, category in enumerate(self.categories):
            data_file_path = os.path.join(self.root, category)
            for file_name in os.listdir(data_file_path):
                file_path = os.path.join(data_file_path, file_name)
                self.images_path.append(file_path)
                self.labels.append(i)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]
        return image, label


if __name__ == "__main__" :
    root = r"C:\CODE\ML - VietNgAI\data\drive_data\animals_v2\animals"
    transform = Compose([
        ToTensor(),
        Resize((264, 264)),
    ])
    x_train = AnimalDataset(root, train=True, transform = transform)
    img, label = x_train.__getitem__(1202)
    x_train_dataLoader = DataLoader(
        dataset=x_train,
        batch_size=16, num_workers=4, shuffle= True, drop_last = False
    )
    epochs = 10
    # print(len(x_train))
    # print(img.shape)
    # cv2.imshow("img", img)
    # print(label)
    # cv2.waitKey(0)

    for image, label in x_train_dataLoader:
        print(image.shape)
        print(label)


