import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import SGD

from tqdm import tqdm
from models import NeuralNetB, NeuralNetA

import matplotlib.pyplot as plt
from typing import Tuple, Union, List, Callable

LEARNING_RATE = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Ms = [300, 400, 500]
M = 500
K = 14
N = 14
BATCH_SIZE = 128
NUM_EPOCHS = 40
MOMENTUM = 0.9

def train(
        model: nn.Module, optimizer: SGD,
        train_loader: DataLoader, val_loader: DataLoader,
        epochs: int = 20
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
  Trains a model for the specified number of epochs using the loaders.

  Returns:
    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
  """

    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for _ in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        for (x_batch, labels) in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (BATCH_SIZE * len(train_loader)))

        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for (v_batch, labels) in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc / (BATCH_SIZE * len(val_loader)))

    return train_losses, train_accuracies, val_losses, val_accuracies


def parameter_search(train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_fn: Callable[[], nn.Module]) -> float:
    """
  Parameter search for our linear model using SGD.

  Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

  Returns:
    The learning rate with the least validation loss.
  """
    num_iter = 10
    best_loss = float('inf')
    best_lr = 0

    lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter)

    for lr in lrs:
        print(f"trying learning rate {lr}")
        for m in Ms:
            print(f'trying {m = }')

            # model = NeuralNetA(M=m)
            model = NeuralNetB(M=m, N=14, K=5, device=DEVICE)

            optim = SGD(model.parameters(), lr, momentum=MOMENTUM)

            train_loss, train_acc, val_loss, val_acc = train(
                model,
                optim,
                train_loader,
                val_loader,
                epochs=NUM_EPOCHS
            )

            if min(val_loss) < best_loss:
                print(f'Min Loss: {val_loss}')
                best_loss = min(val_loss)
                print(f'Best Learning Rate: {lr} ')
                best_lr = lr

    return best_lr


def evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (batch, labels) in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (BATCH_SIZE * len(loader))
        return test_loss, test_acc


def main():
    train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True,
                                                 transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                                transform=torchvision.transforms.ToTensor())

    train_dataset, val_dataset = random_split(train_dataset,
                                              [int(0.9 * len(train_dataset)), int(0.1 * len(train_dataset))])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    best_lr = parameter_search(train_loader, val_loader, NeuralNetB(M, K, N, DEVICE))

    model1B = NeuralNetB(M=M, K=K, N=N, device=DEVICE)
    optimizer_1B = SGD(model1B.parameters(), best_lr, momentum=MOMENTUM)

    model2B = NeuralNetB(M=M, K=K, N=N, device=DEVICE)
    optimizer_2B = SGD(model2B.parameters(), 0.027, momentum=MOMENTUM)

    model3B = NeuralNetB(M=300, K=K, N=N, device=DEVICE)
    optimizer_3B = SGD(model3B.parameters(), 0.03, momentum=MOMENTUM)

    model4B = NeuralNetB(M=400, K=K, N=N, device=DEVICE)
    optimizer_4B = SGD(model4B.parameters(), 0.03, momentum=MOMENTUM)

    model5B = NeuralNetB(M=400, K=K, N=N, device=DEVICE)
    optimizer_5B = SGD(model5B.parameters(), 0.027, momentum=MOMENTUM)

    train_loss_bestB, train_accuracy_bestB, val_loss_bestB, val_accuracy_bestB = train(
        model1B, optimizer_1B, train_loader, val_loader, NUM_EPOCHS)

    train_loss_2ndB, train_accuracy_2ndB, val_loss_2ndB, val_accuracy_2ndB = train(
        model2B, optimizer_2B, train_loader, val_loader, NUM_EPOCHS)

    train_loss_3rdB, train_accuracy_3rdB, val_loss_3rdB, val_accuracy_3rdB = train(
        model3B, optimizer_3B, train_loader, val_loader, NUM_EPOCHS)

    train_loss_4thB, train_accuracy_4thB, val_loss_4thB, val_accuracy_4thB = train(
        model4B, optimizer_4B, train_loader, val_loader, NUM_EPOCHS)

    train_loss_5thB, train_accuracy_5thB, val_loss_5thB, val_accuracy_5thB = train(
        model5B, optimizer_5B, train_loader, val_loader, NUM_EPOCHS)

    """Plot the training and validation accuracy for each epoch."""

    epochs = range(1, 41)
    plt.plot(epochs, train_accuracy_bestB, label="Train Accuracy Best")
    plt.plot(epochs, val_accuracy_bestB, label="Validation Accuracy Best")

    plt.plot(epochs, train_accuracy_2ndB, label="Train Accuracy 2nd")
    plt.plot(epochs, val_accuracy_2ndB, label="Validation Accuracy 2nd")

    plt.plot(epochs, train_accuracy_3rdB, label="Train Accuracy 3rd")
    plt.plot(epochs, val_accuracy_3rdB, label="Validation Accuracy 3rd")

    plt.plot(epochs, train_accuracy_4thB, label="Train Accuracy 4th")
    plt.plot(epochs, val_accuracy_4thB, label="Validation Accuracy 4th")

    plt.plot(epochs, train_accuracy_5thB, label="Train Accuracy 5th")
    plt.plot(epochs, val_accuracy_5thB, label="Validation Accuracy 5th")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Logistic Regression Accuracy for CIFAR-10 vs Epoch")

    test_loss, test_acc = evaluate(model1B, test_loader)
    print(f"Test Accuracy: {test_acc}")
