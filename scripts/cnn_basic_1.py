import torch
import pandas as pd
import numpy as np
from torch import nn
from PIL import Image
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader

prefix = "/groups/CS156b/data/"
classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices']

train = "/groups/CS156b/data/student_labels/train.csv"
traindf = pd.read_csv(train)

test = "/groups/CS156b/data/student_labels/test_ids.csv"
testdf = pd.read_csv(test)

# nans as -1
classesdf = traindf[classes].fillna(-1).iloc[:1000]

paths = traindf["Path"].iloc[:1000].tolist()

# most seem to be 2320, 2828, but smaller for now
Xdf = np.array([np.asarray(Image.open(prefix+path).resize((50, 50))) for path in paths])
X_train = torch.from_numpy(Xdf.reshape((-1, 1, 50, 50)).astype('float32'))

y_train = torch.from_numpy((classesdf+1).to_numpy().reshape((-1, 14, 1)).astype('float32'))
train_dataset = TensorDataset(X_train, y_train)
training_data_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=(3,3)),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.5),

    nn.Conv2d(8, 8, kernel_size=(3,3)),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Dropout(p=0.5),

    nn.Flatten(),
    nn.Linear(968, 64),
    nn.ReLU(),
    nn.Linear(64, 14)
    # PyTorch implementation of cross-entropy loss includes softmax layer
)

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters())

# Train the model for 10 epochs, iterating on the data in batches
n_epochs = 10

# store metrics
training_accuracy_history = np.zeros([n_epochs, 1])
training_loss_history = np.zeros([n_epochs, 1])
validation_accuracy_history = np.zeros([n_epochs, 1])
validation_loss_history = np.zeros([n_epochs, 1])

for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}/10:', end='')
    train_total = 0
    train_correct = 0
    # train
    model.train()
    for i, data in enumerate(training_data_loader):
        images, labels = data
        optimizer.zero_grad()
        # forward pass
        output = model(images)
        # calculate categorical cross entropy loss
        loss = criterion(output, labels)
        # backward pass
        loss.backward()
        optimizer.step()
        # track training accuracy
        _, predicted = torch.max(output.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        # track training loss
        training_loss_history[epoch] += loss.item()
        # progress update after 180 batches (~1/10 epoch for batch size 32)
        if i % 180 == 0: print('.',end='')
    training_loss_history[epoch] /= len(training_data_loader)
    training_accuracy_history[epoch] = train_correct / train_total
    print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}, acc: {training_accuracy_history[epoch,0]:0.4f}',end='')

# out.insert(0, 'Id', testdf['Id'])
# out.to_csv("CS156b/zeros.csv", index=False)
