import os.path

import torch
from torch import nn
from CIFAIR import MyDataset
from TryCreateMyDataset import AnimalDataset
from model import SimpleCNN
from torch.utils.data import DataLoader
from torch.optim import SGD
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torchvision.transforms import ToTensor, Compose, Resize
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import shutil
from tranfer_learning import MyResNet

def get_args():
    parser = ArgumentParser(description='Train a CNN model')
    parser.add_argument("--epochs", "-e", type=int, default=100 , help="number of epochs")
    parser.add_argument("--batch_size", '-b', type=int, default=64, help="batch size")
    parser.add_argument("--logging", "-l", type=str, default="tensorboard", help="logging level")
    parser.add_argument("--train_model", "-train", type=str, default="train_model", help="train_model")
    parser.add_argument("--checkpoint", "-c", type=str, default=None)
    args = parser.parse_args()
    return args

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="ocean")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)

if __name__ == '__main__':
    args = get_args()
    if os.path.isdir(args.logging):
        shutil.rmtree(args.logging)
    summary_writer = SummaryWriter(args.logging)

    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transform ảnh
    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
    ])

    # Dữ liệu
    root = r"C:\CODE\ML - VietNgAI\data\drive_data\animals_v2\animals"
    train_dataset = AnimalDataset(root=root, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=4)
    test_dataset = AnimalDataset(root=root, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=4)

    # Mô hình
    model = MyResNet().to(device)  # Di chuyển model lên GPU

    # Chechpoint model
    if not os.path.isdir(args.train_model):
        os.mkdir(args.train_model)

    # Loss function & Optimizer
    for name, param in model.named_parameters():
        if "fc" in name or "layer4" in name:
            pass
        else:
            param.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        start_epochs = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint["best_acc"]
    else:
        start_epochs = 0
        best_acc = 0

    # Huấn luyện
    for epoch in range(start_epochs, args.epochs):
        model.train()
        progress_bar = tqdm(train_loader)
        for iter, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device)  # Di chuyển data lên GPU
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            summary_writer.add_scalar("train_loss", loss, iter + epoch * len(train_loader))
            loss.backward()
            optimizer.step()

            progress_bar.set_description(f"epoch_{epoch+1}/{args.epochs}, Iteration_{iter+1}/{len(train_loader)}, Loss_{loss.item()}")

        # Đánh giá

        model.eval()
        all_labels, all_preds = [], []
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # Di chuyển data lên GPU
            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())  # Chuyển về CPU để tránh lỗi
            all_preds.extend(preds.cpu().numpy())
        result = accuracy_score(all_labels, all_preds)
        print(result)
        summary_writer.add_scalar("result", result, epoch)
        plot_confusion_matrix(summary_writer, confusion_matrix(all_labels, all_preds), class_names=test_dataset.categories, epoch=epoch)
        checkpoint = {
            "epoch": epoch+1,
            "model": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(args.train_model, "checkpoint.pt"))
        if best_acc < result:
            checkpointBest = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "best_acc": result,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpointBest, os.path.join(args.train_model, "cnnAnimal_best.pt"))
            best_acc = result
        summary_writer.flush()