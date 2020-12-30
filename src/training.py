"""Training pipeline."""
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from .model import Net
from .data import MNISTDataset, transformer

EPOCHS = 16
BATCH_SIZE = 128


def train():
    """Training pipeline."""
    # CUDA + Tensorboard
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tensorboard_writer = SummaryWriter()

    # Split train val
    train_df, val_df = train_test_split(
        pd.read_csv(os.path.join("data", "train.csv")),
        train_size=0.8,
        random_state=42
    )

    # Create 2 datasets
    dataset_train = MNISTDataset(train_df, transformer)
    dataset_val = MNISTDataset(val_df)

    # Dataloaders
    trainloader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE
    )
    valloader = torch.utils.data.DataLoader(
        dataset_val, batch_size=BATCH_SIZE
    )

    model = Net().float().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):  # loop over the dataset multiple times
        model.train()
        loss_train = 0.0
        correct_train = 0
        # Training
        for i, data in enumerate(tqdm(trainloader, desc=f"Training epoch {epoch}"), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Compute accuracy
            correct_train += (
                outputs.argmax(dim=1) == labels
            ).int().sum().item()

            # print statistics
            loss_train += loss.item()

        # Validation
        model.eval()
        loss_val = 0.0
        correct_val = 0
        with torch.no_grad():
            for i, data in enumerate(tqdm(valloader, desc=f"Validation epoch {epoch}"), 0):
                inputs, labels = data
                inputs = inputs.float().to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Compute accuracy
                correct_val += (
                    outputs.argmax(dim=1) == labels
                ).int().sum().item()

                # print statistics
                loss_val += loss.item()

        # Add the loss
        tensorboard_writer.add_scalar(
            "Loss/Train", loss_train, epoch
        )
        tensorboard_writer.add_scalar(
            "Loss/Val", loss_val, epoch
        )

        # Add Accuracy
        tensorboard_writer.add_scalar(
            "Accuracy/Train", correct_train /
            (len(trainloader)*BATCH_SIZE), epoch
        )
        tensorboard_writer.add_scalar(
            "Accuracy/Val", correct_val /
            (len(valloader)*BATCH_SIZE), epoch
        )
    print('Finished Training')

    # Save the model
    torch.save(model, os.path.join('model.pt'))


if __name__ == "__main__":
    train()
