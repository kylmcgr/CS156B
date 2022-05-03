import cv2
import numpy as np
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
OUTPUT_PATH = "/home/mcgee/CS156b/preprocess/"


def gen_cnn_basic():
    model = nn.Sequential(
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
        nn.Linear(64, 14),
        nn.Tanh(),
    )
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters())

    return (model, criterion, optimizer)


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
        nn.Tanh(),
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    return (model, criterion, optimizer)


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
        nn.Tanh(),
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return (model, criterion, optimizer)


def canny_edge_detection(img_name):
    # file_name = os.path.join(os.path.dirname(__file__), img_name)

    # Read the original image
    img = cv2.imread(img_name, -1)
    img = cv2.resize(img, (320, 320))
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


def edge_detection_test(model, criterion, optimizer, output_path):
    print("Testing edge detection.")

    traindf = pd.read_csv(TRAIN_PATH)
    traindf = traindf.iloc[:1000]
    classesdf = traindf[PATHOLOGIES]
    paths = traindf.loc[traindf["Path"].str.startswith("train/"), "Path"]

    # most seem to be 2320, 2828, but smaller for now
    print("Getting data...")
    original = []
    edges = []

    for _i, path in paths.iteritems():
        original.append(
            np.asarray(Image.open(DATA_PATH + path).resize((320, 320)))
        )
        edges.append(np.asarray(canny_edge_detection(DATA_PATH + path)))

    original = np.array(original)
    edges = np.array(edges)

    for name, images in {"original": original, "edges": edges}:
        f = open(f"{output_path}_{name}.csv", "w")
        X_train = torch.from_numpy(
            images.reshape((-1, 1, 320, 320)).astype("float32")
        )

        imputer = SimpleImputer(
            missing_values=np.nan, strategy="constant", fill_value=0
        )
        # imputer =  SimpleImputer(missing_values=np.nan, strategy="mean")
        imputed_data = imputer.fit_transform(classesdf)
        # f.write(f"classesdf: {imputed_data}\n")

        y_train = torch.from_numpy((imputed_data + 1).astype("float32"))
        # f.write(f"ytrain: {y_train}\n")

        train_dataset = TensorDataset(X_train, y_train)
        training_data_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=False
        )

        device = torch.device("cuda:0")

        model.to(device)

        # Train the model for n epochs, iterating on the data in batches
        n_epochs = 10

        # store metrics
        training_loss_history = np.zeros([n_epochs, 1])

        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}:", end="")
            f.write(f"Epoch {epoch+1}/{n_epochs}:")

            # train
            model.train()
            for i, data in enumerate(training_data_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                # forward pass
                output = model(images)
                # f.write(f"output: {output}")

                # calculate categorical cross entropy loss
                loss = criterion(output, labels)
                # backward pass
                loss.backward()
                optimizer.step()
                # track training loss
                training_loss_history[epoch] += loss.item()
                # progress update after 180 batches (~1/10 epoch for batch
                # size 32)
                if i % 180 == 0:
                    print(".", end="")
            training_loss_history[epoch] /= len(training_data_loader)
            print(f"\n\tloss: {training_loss_history[epoch,0]:0.4f}", end="")
            f.write(f"\n\tloss: {training_loss_history[epoch,0]:0.4f}\n")

    # write training_loss (or just get training loss over epochs)
    f.close()


# (model, criterion, optimizer)
(model, criterion, optimizer) = gen_cnn_basic()
edge_detection_test(
    model, criterion, optimizer, OUTPUT_PATH + "cnn_basic_edge_detection"
)

(model, criterion, optimizer) = gen_cnn_resnet()
edge_detection_test(
    model, criterion, optimizer, OUTPUT_PATH + "cnn_resnet_edge_detection"
)

(model, criterion, optimizer) = gen_cnn_densenet()
edge_detection_test(
    model, criterion, optimizer, OUTPUT_PATH + "cnn_densenet_edge_detection"
)
