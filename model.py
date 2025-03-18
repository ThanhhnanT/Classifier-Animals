import torch
import torch.nn as nn

class SimpleNeuralNetWork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()   # Khoi tao init cha
        self.platten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=3072, out_features=256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1024),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=num_classes),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.platten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = self.__make_block(in_channels =3, out_channels =8)
        self.conv2 = self.__make_block(in_channels =8, out_channels =16)
        self.conv3 = self.__make_block(in_channels =16, out_channels =32)
        self.conv4 = self.__make_block(in_channels =32, out_channels =64)
        self.conv5 = self.__make_block(in_channels =64, out_channels =128)

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features= 2048, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features= 512, out_features=num_classes),
        )

    def __make_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x= self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

if __name__ == '__main__':
    model = SimpleCNN()
    input = torch.randn(8, 3, 264, 268)
    output = model(input)
    print(output)