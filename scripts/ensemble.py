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

def load_traindata(partialData=False, numdata=1000, imagex=320, imagey=320):
	prefix = "/groups/CS156b/data/"
	train = "/groups/CS156b/data/student_labels/train.csv"
	traindf = pd.read_csv(train)
	classesdf = traindf[classes][:-1]
	paths = traindf["Path"].tolist()[:-1]
	if partialData:
		classesdf = traindf[classes].iloc[:numdata]
		paths = traindf["Path"].iloc[:numdata].tolist()
	Xdf = np.array([np.asarray(Image.open(prefix+path).resize((imagex, imagey))) for path in paths])
	return Xdf, classesdf

def load_testdata(partialData=False, numtest=10, imagex=320, imagey=320):
	prefix = "/groups/CS156b/data/"
	test = "/groups/CS156b/data/student_labels/test_ids.csv"
	testdf = pd.read_csv(test)
	testpaths = testdf["Path"].tolist()
	if partialData:
		testpaths = testdf["Path"].iloc[:numtest].tolist()
	Xtestdf = np.array([np.asarray(Image.open(prefix+path).resize((imagex, imagey))) for path in testpaths])
	X_test = torch.from_numpy(Xtestdf.reshape((-1, 1, imagex, imagey)).astype('float32'))
	ids = testdf['Id'].tolist()
	if partialData:
		ids = testdf['Id'].iloc[:numtest].tolist()
	return X_test, ids

def get_CNN(device, updateWeights=False):
	model = nn.Sequential(
	    nn.Conv2d(1, 64, kernel_size=(3,3)),
	    nn.ReLU(),
	    nn.MaxPool2d(2),
	    nn.Dropout(p=0.5),

	    nn.Conv2d(64, 64, kernel_size=(3,3)),
	    nn.ReLU(),
	    nn.MaxPool2d(2),
	    nn.Dropout(p=0.5),

	    nn.Conv2d(64, 128, kernel_size=(3,3)),
	    nn.ReLU(),
	    nn.MaxPool2d(2),
	    nn.Dropout(p=0.5),

	    nn.Conv2d(128, 128, kernel_size=(3,3)),
	    nn.ReLU(),
	    nn.MaxPool2d(2),
	    nn.Dropout(p=0.5),

	    nn.Flatten(),
	    nn.Linear(25088, 3456),
	    nn.ReLU(),
	    nn.Dropout(0.2),
	    nn.Linear(3456, 288),
	    nn.ReLU(),
	    nn.Dropout(0.2),
	    nn.Linear(288, 64),
	    nn.ReLU(),
	    nn.Linear(64, 1),
	    nn.Tanh()
	)
	return model

def get_densenet(device, updateWeights=False):
	model = models.densenet161(pretrained=True)
	model.features.conv0 = nn.Conv2d(1, 96, kernel_size=7, stride=2, padding=3,bias=False)
	for param in model.parameters():
	    param.requires_grad = updateWeights
	model.classifier = nn.Sequential(nn.Linear(2208, 512),
	                                 nn.ReLU(),
	                                 nn.Dropout(0.2),
	                                 nn.Linear(512, 1),
	                                 nn.LogSoftmax(dim=1),
	                                 nn.Tanh())
	return model

def get_inception(device, updateWeights=False):
	model = models.inception_v3(pretrained=True)
	model.transform_input = False
	model.Conv2d_1a_3x3 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, bias=False)
	model.fc = nn.Linear(2048, 1)
	return model

def get_resnet(device, updateWeights=False):
	model = models.resnet50(pretrained=True)
	model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
	for param in model.parameters():
	    param.requires_grad = False
	model.fc = nn.Sequential(nn.Linear(2048, 512),
	                                 nn.ReLU(),
	                                 nn.Dropout(0.2),
	                                 nn.Linear(512, 1),
	                                 nn.LogSoftmax(dim=1),
	                                 nn.Tanh())
	return model

def get_vgg(device, updateWeights=False):
	model = models.vgg16(pretrained=True)
	model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
	model.classifier[6] = nn.Linear(4096, 1)
	return model

def fit_model(model, training_data_loader, device, n_epochs=20):
	# criterion = nn.MSELoss()
# 	criterion = nn.NLLLoss()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	model.to(device)
	training_loss_history = np.zeros([n_epochs, 1])
	for epoch in range(n_epochs):
	    print(f'Epoch {epoch+1}/{n_epochs}:', end='')
	    model.train()
	    for i, data in enumerate(training_data_loader):
	        images, labels = data
	        images, labels = images.to(device), labels.to(device)
	        optimizer.zero_grad()
	        output = model.forward(images)
	        loss = criterion(output, labels)
	        loss.backward()
	        optimizer.step()
	        training_loss_history[epoch] += loss.item()
	        if i % 180 == 0: print('.',end='')
	    training_loss_history[epoch] /= len(training_data_loader)
	    print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}',end='')
	return model

def test_model(classes, test_data_loader, filename, ids):
	out = np.empty((0,len(classes)), int)
	with torch.no_grad():
	    model.eval()
	    for i, data in enumerate(test_data_loader):
	        images = data[0].to(device)
	        output = model(images).cpu().numpy()
	        out = np.append(out, output, axis=0)
	outdf = pd.DataFrame(data = out, columns=classes)
	outdf.insert(0, 'Id', ids)
	outdf.to_csv(filename, index=False)

if __name__ == "__main__":
	classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
	        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
	        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
	        'Pleural Other', 'Fracture', 'Support Devices']
	# groups = [['Enlarged Cardiomediastinum', 'Cardiomegaly'],
	#         ['Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
	#         'Pneumonia', 'Atelectasis'], ['Pneumothorax', 'Pleural Effusion',
	#         'Pleural Other'], ['No Finding', 'Fracture', 'Support Devices']]
	filename = "/home/kmcgraw/CS156b/predictions/emseble_densenet_CE_256x256.csv"
	batch_size = 64
	imagex, imagey = 256, 256
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	Xdf, classesdf = load_traindata(imagex=imagex, imagey=imagey)
	X_test, ids = load_testdata(imagex=imagex, imagey=imagey)
	for i in range(len(classes)):
		model = get_densenet(device)
		knownValues = ~classesdf[classes[i]].isna()
		x_vals = Xdf[knownValues]
		y_vals = classesdf[classes[i]].loc[knownValues]
		X_train = torch.from_numpy(x_vals.reshape((-1, 1, imagex, imagey)).astype('float32'))
		y_train = torch.from_numpy(y_vals.to_numpy().astype('float32'))
		train_dataset = TensorDataset(X_train, y_train)
		training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
		trained_model = fit_model(model, training_data_loader, device)
		test_dataset = TensorDataset(X_test)
		test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
		test_model(classes, test_data_loader, filename, ids)
