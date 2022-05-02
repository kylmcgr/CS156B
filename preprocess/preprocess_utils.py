import cv2
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from torch import optim
from torchvision import models
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
from sklearn.impute import SimpleImputer

DATA_PATH = "/groups/CS156b/data/"
PATHOLOGIES = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
TRAIN_PATH = "/groups/CS156b/data/student_labels/train.csv"
TEST_PATH = "/groups/CS156b/data/student_labels/test_ids.csv"


def gen_cnn_basic():
    return nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.5),
        nn.Conv2d(64, 64, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.5),
        nn.Conv2d(64, 128, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.5),
        nn.Conv2d(128, 128, kernel_size=(3, 3)),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.5),
        nn.Flatten(),
        nn.Linear(41472, 3456),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(3456, 288),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(288, 64),
        nn.ReLU(),
        nn.Linear(64, 14)
        # PyTorch implementation of cross-entropy loss includes softmax layer
    )


def gen_cnn_resnet():
    model = models.resnet50(pretrained=True)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 14),
        nn.LogSoftmax(dim=1),
    )
    return model


def gen_cnn_densenet():
    model = models.densenet161(pretrained=True)
    model.features.conv0 = nn.Conv2d(
        1, 96, kernel_size=7, stride=2, padding=3, bias=False
    )

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(2208, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 14),
        nn.LogSoftmax(dim=1),
    )

    return model


def train_model(model, traindf, classesdf, output_path):
    f = open(output_path, "w")

    paths = traindf["Path"].tolist()

    # most seem to be 2320, 2828, but smaller for now
    print("Getting data...")
    Xdf = np.array(
        [
            np.asarray(Image.open(DATA_PATH + path).resize((320, 320)))
            for path in paths
        ]
    )
    X_train = torch.from_numpy(
        Xdf.reshape((-1, 1, 320, 320)).astype("float32")
    )

    y_train = torch.from_numpy((classesdf + 1).to_numpy().astype("float32"))

    train_dataset = TensorDataset(X_train, y_train)
    training_data_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False
    )

    device = torch.device("cuda:0")

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters())

    model.to(device)

    # Train the model for 10 epochs, iterating on the data in batches
    n_epochs = 10

    # store metrics
    training_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/10:", end="")
        f.write(f"Epoch {epoch+1}/10:")

        # train
        model.train()
        for i, data in enumerate(training_data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward pass
            output = model(images)
            # calculate categorical cross entropy loss
            loss = criterion(output, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            # track training loss
            training_loss_history[epoch] += loss.item()
            # progress update after 180 batches (~1/10 epoch for batch size 32)
            if i % 180 == 0:
                print(".", end="")
        training_loss_history[epoch] /= len(training_data_loader)
        print(f"\n\tloss: {training_loss_history[epoch,0]:0.4f}", end="")
        f.write(f"\n\tloss: {training_loss_history[epoch,0]:0.4f}")

    # write training_loss (or just get training loss over epochs)
    f.close()


def imputation_test(model, output_path):
    print("Testing imputation.")
    traindf = pd.read_csv(TRAIN_PATH)
    classesdf = traindf[PATHOLOGIES]

    impute_mehtods = {
        "-1": SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value=-1
        ),
        "0": SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value=0
        ),
        "mean": SimpleImputer(
            missing_values=np.nan,
            strategy="mean",
        ),
    }

    for name, imputer in impute_mehtods.items():
        print(f"Trying Imputation with: {name}")

        imputer.fit_transform(classesdf)
        train_model(model, traindf, classesdf, f"{output_path}_{name}.csv")


def sobel_edge_detection(img_name):
    file_name = os.path.join(os.path.dirname(__file__), img_name)

    # Read the original image
    img = cv2.imread(file_name, -1)

    # Display original image
    # cv2.imshow("Original", img)

    # Convert to graycsale
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Sobel Edge Detection
    sobelxy = cv2.Sobel(
        src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5
    )  # Combined X and Y Sobel Edge Detection
    # cv2.imshow("Sobel X Y using Sobel() function", sobelxy)

    return sobelxy


def canny_edge_detection(img_name):
    file_name = os.path.join(os.path.dirname(__file__), img_name)

    # Read the original image
    img = cv2.imread(file_name, -1)
    # Display original image
    # cv2.imshow("Original", img)

    # Convert to graycsale
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)

    # Canny Edge Detection
    edges = cv2.Canny(
        image=img_blur, threshold1=100, threshold2=200
    )  # Canny Edge Detection
    # Display Canny Edge Detection Image
    # cv2.imshow("Canny Edge Detection", edges)

    return edges


def edge_detection_test(model, type, output_path):
    f = open(output_path, "w")

    traindf = pd.read_csv(TRAIN_PATH)
    classesdf = traindf[PATHOLOGIES]

    imputer = SimpleImputer(missing_value=np.nan, strategy=-1)
    imputer.fit_transform(classesdf)

    paths = traindf["Path"].tolist()

    # most seem to be 2320, 2828, but smaller for now
    # TODO: transform image
    Xdf = np.array(
        [
            np.asarray(Image.open(DATA_PATH + path).resize((320, 320)))
            for path in paths
        ]
    )
    X_train = torch.from_numpy(
        Xdf.reshape((-1, 1, 320, 320)).astype("float32")
    )

    y_train = torch.from_numpy((classesdf + 1).to_numpy().astype("float32"))

    train_dataset = TensorDataset(X_train, y_train)
    training_data_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=False
    )

    device = torch.device("cuda:0")

    model = model

    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters())

    model.to(device)

    # Train the model for 10 epochs, iterating on the data in batches
    n_epochs = 10

    # store metrics
    training_loss_history = np.zeros([n_epochs, 1])

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/10:", end="")
        f.write(f"Epoch {epoch+1}/10:")

        # train
        model.train()
        for i, data in enumerate(training_data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward pass
            output = model(images)
            # calculate categorical cross entropy loss
            loss = criterion(output, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            # track training loss
            training_loss_history[epoch] += loss.item()
            # progress update after 180 batches (~1/10 epoch for batch size 32)
            if i % 180 == 0:
                print(".", end="")
        training_loss_history[epoch] /= len(training_data_loader)
        print(f"\n\tloss: {training_loss_history[epoch,0]:0.4f}", end="")
        f.write(f"\n\tloss: {training_loss_history[epoch,0]:0.4f}")

    # write training_loss (or just get training loss over epochs)
    f.close()


if __name__ == "__main__":
    sobel_edge_detection("img1.jpg")
    canny_edge_detection("img1.jpg")
