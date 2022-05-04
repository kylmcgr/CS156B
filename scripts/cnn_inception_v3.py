import torch
import pandas as pd
import numpy as np
from torch import nn
from PIL import Image
from torch import optim
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

batch_size = 64
resizex = 320
resizey = 320
n_epochs = 10

# n = 250
# n_test = 10

train = "/groups/CS156b/data/student_labels/train.csv"
traindf = pd.read_csv(train)
classesdf = traindf[classes].fillna(-1) #.iloc[:n]
paths = traindf["Path"].tolist()[:-1] # .iloc[:n] [:-1]

Xdf = np.array([np.asarray(Image.open(prefix+path).resize((resizex, resizey))) for path in paths])
X_train = torch.from_numpy(Xdf.reshape((-1, 1, resizex, resizey)).astype('float32'))
y_train = torch.from_numpy((classesdf+1).to_numpy().astype('float32')[:-1]) #[:-1]

train_dataset = TensorDataset(X_train, y_train)
training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0")
model = models.inception_v3(pretrained=True)
model.transform_input = False
model.Conv2d_1a_3x3 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, bias=False)
model.fc = nn.Linear(2048, 14) 
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)

#print(model)

training_loss_history = np.zeros([n_epochs, 1])
for epoch in range(n_epochs):
    print(f'\nEpoch {epoch+1}/{n_epochs}:', end='')
    model.train()
    for i, data in enumerate(training_data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(images)
        # print("\n")
        # print(output.shape)
        # print("\n")
        # print(labels.shape)
        # print("\n")
        # print(images.shape)
        # print("\n")
        loss = criterion(output.logits, labels)
        loss.backward()
        optimizer.step()
        training_loss_history[epoch] += loss.item()
        if i % 180 == 0: print('.', end='')
    training_loss_history[epoch] /= len(training_data_loader)
    print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}',end='')
    
test = "/groups/CS156b/data/student_labels/test_ids.csv"
testdf = pd.read_csv(test)
testpaths = testdf["Path"].tolist() # .iloc[:n_test]

Xtestdf = np.array([np.asarray(Image.open(prefix+path).resize((resizex, resizey))) for path in testpaths])
X_test = torch.from_numpy(Xtestdf.reshape((-1, 1, resizex, resizey)).astype('float32'))

test_dataset = TensorDataset(X_test)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

out = np.empty((0,len(classes)), int)
with torch.no_grad():
    model.eval()
    for i, data in enumerate(test_data_loader):
        images = data[0].to(device)
        output = model(images).cpu().numpy()
        out = np.append(out, output, axis=0)
        
outdf = pd.DataFrame(data = out, columns=traindf.columns[6:])
outdf.insert(0, 'Id', testdf['Id'].tolist()) # .iloc[:n_test]
outdf.to_csv("/home/bjuarez/CS156b/predictions/cnn_inception_v3_initial_320x320.csv", index=False)