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

# n = 250
# n_test = 10

classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices']

batch_size = 64
resizex = 320
resizey = 320
n_epochs = 20

traindf = pd.read_csv("/groups/CS156b/data/student_labels/train.csv")
classesdf = traindf[classes].fillna(0) #.iloc[:n] -1 -> 0
paths = traindf["Path"].tolist()[:-1] # .iloc[:n] [:-1]

Xdf = np.array([np.asarray(Image.open("/groups/CS156b/data/"+path).resize((resizex, resizey))) for path in paths])
X_train = torch.from_numpy(Xdf.reshape((-1, 1, resizex, resizey)).astype('float32'))
y_train = torch.from_numpy((classesdf+1).to_numpy().astype('float32')[:-1]) #[:-1]

train_dataset = TensorDataset(X_train, y_train)
training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda:0")

model = models.googlenet(pretrained=True)

model.transform_input = False

model.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, bias=False)
# for param in model.parameters():
#     param.requires_grad = False
model.fc = nn.Sequential(
    nn.Linear(2048, 14),
    nn.Tanh())
    
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.to(device)

training_loss_history = np.zeros([n_epochs, 1])
for epoch in range(n_epochs):
    print(f'\nEpoch {epoch+1}/{n_epochs}:', end='')
    model.train()
    for i, data in enumerate(training_data_loader):
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output.logits, labels)
        loss.backward()
        optimizer.step()
        training_loss_history[epoch] += loss.item()
        # if i % 180 == 0: print('.', end='')
    training_loss_history[epoch] /= len(training_data_loader)
    print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}',end='')
    
testdf = pd.read_csv("/groups/CS156b/data/student_labels/test_ids.csv")
testpaths = testdf["Path"].tolist() # .iloc[:n_test]

Xtestdf = np.array([np.asarray(Image.open("/groups/CS156b/data/"+path).resize((resizex, resizey))) for path in testpaths])
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
outdf.to_csv("/home/bjuarez/CS156b/predictions/cnn_googlenet_320x320_addedTanh_fill0_unF.csv", index=False)