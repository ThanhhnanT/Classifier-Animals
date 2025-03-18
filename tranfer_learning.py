from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn as nn
from torchsummary import summary

class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(weights = ResNet50_Weights.DEFAULT)
        self.model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=512, out_features=10),
        )

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device is {}".format(device))
    image = torch.randn(1, 3, 224, 224).to(device)
    model = MyResNet().to(device)
    print(model)
    for name, param in model.named_parameters():
        if "fc" in name or "layer4" in name:
            pass
        else:
            param.requires_grad = False
        print(name, param.requires_grad)
    summary(model, image)
    # output = model(image)
    # print(output.shape)