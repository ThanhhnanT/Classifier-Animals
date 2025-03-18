from argparse import ArgumentParser
from model import SimpleCNN
import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor, Resize, Compose
from torchsummary import summary
from tranfer_learning import MyResNet
import torch.nn as nn


def get_args():
    parser = ArgumentParser(description = "CNN interface")
    parser.add_argument("--image-size", "-i", type =int, default = 224, help = "image size")
    parser.add_argument("--image-path", "-p", type =str, default = None)
    parser.add_argument("--checkpoint", "-c", type=str, default = "train_model/cnnAnimal_best.pt")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    model = MyResNet().to(device)
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
    else:
        print("No checkpoint given")
        exit(0)

    origin_image = cv2.imread(args.image_path)
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
    print(type(origin_image))
    transform = Compose(
        [
            ToTensor(),
            Resize((224, 224)),
        ]
    )
    image = transform(origin_image)
    print(image.shape)
    image = image.unsqueeze(0)
    print(image.shape)
    image = image.to(device)
    model.eval()
    model.to(device)
    result = model(image)
    result = nn.Softmax(dim=1)(result)
    pred_index = torch.argmax(result)
    print(pred_index)
    category=["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    origin_image = cv2.cvtColor(origin_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("{}/{}%".format(category[pred_index], result[0, pred_index]*100), origin_image)

    cv2.waitKey(0)
