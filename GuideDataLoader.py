from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from TryCreateMyDataset import MyDataset

if __name__ =="__main__":
    root = r"C:\CODE\ML - VietNgAI\data\drive_data\animals_v2\animals"
    training_data = CIFAR10(root='data', train=True, download=True, transform=ToTensor())
    # training_data = MyDataset(root = root, train=True)
    # image, label = training_data.__getitem__(1234)
    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=16,  # Số lượng bức ảnh lấy ra cùng 1 lúc
        num_workers=4, # Số nhân để làm việc
        shuffle= True, # Tráo bộ dữ liệu, lấy ngẫu nhiên
        drop_last = False # Khi số lượng không chia hết cho batch_size thì bỏ đi ảnh dư
    )
    for image, label in train_dataloader:
        print(image.shape)
        print(label)








