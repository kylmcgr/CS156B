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
import cv2
import skimage.io
import skimage.color
import skimage.filters
from sklearn import preprocessing


def preprocessing_complex(image):
	new_image = image.resize((320, 320))
	new_image = np.float32(new_image)
	# Bilateral filter
	bilateral = cv2.bilateralFilter(new_image, 5, 50, 50)
	# adaptiveThresh = cv2.adaptiveThreshold(img_mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
	#                                           cv2.THRESH_BINARY, 199, 5)
	# adaptiveThresh = skimage.filters.threshold_otsu(new_image)
	block_size =75
	local_thresh = skimage.filters.threshold_local(new_image, block_size, offset=5)
	binary_local = new_image > local_thresh
	gaussHist = skimage.exposure.equalize_hist(new_image)
	# gray_image = skimage.color.rgb2gray(bilateral)
	# blurring may not be required
	# blurred_image = skimage.filters.gaussian(bilateral, sigma=1.0)
	# find max and min pixel intensities, create threshold value -- need to alter to be shorter
	max = 0
	min = new_image[0,0]
	for i in range(new_image.shape[0]):
	    for j in range(new_image.shape[1]):
	        if new_image[i,j] > max:
	            max = new_image[i,j]
	        if new_image[i,j] < min:
	            min = new_image[i,j]
	t = min + 0.9 * (max-min)
	# create a mask based on the threshold
	binary_mask = new_image < t
	# closing mask, removing small areas
	area_closed = skimage.morphology.area_closing(binary_mask,area_threshold = 128)
	total_img = np.stack([gaussHist,binary_local,bilateral],axis=-1)
	selection = total_img.copy()
	selection[~area_closed] = 0
	return selection

def preprocessing_simple(image):
	new_image = image.resize((320, 320))
	new_image = np.float32(new_image)
	# Bilateral filter
	bilateral = cv2.bilateralFilter(new_image, 5, 50, 50)
	# blurring may not be required
	blurred_image = skimage.filters.gaussian(bilateral, sigma=1.0)
	# find max and min pixel intensities, create threshold value -- need to alter to be shorter
	max = 0
	min = blurred_image[0,0]
	for i in range(blurred_image.shape[0]):
	    for j in range(blurred_image.shape[1]):
	        if blurred_image[i,j] > max:
	            max = blurred_image[i,j]
	        if blurred_image[i,j] < min:
	            min = blurred_image[i,j]
	t = min + 0.9 * (max-min)
	# create a mask based on the threshold
	binary_mask = blurred_image < t
	# closing mask, removing small areas
	area_closed = skimage.morphology.area_closing(binary_mask,area_threshold = 128)
	# use the binary_mask to select the "interesting" part of the image
	selection = blurred_image.copy()
	selection[~area_closed] = 0
	return selection

def load_traindata(processing, classes, partialData=False, numdata=1000, fillna=True, naVal=-1, imagex=320, imagey=320):
	prefix = "/groups/CS156b/data/"
	train = "/groups/CS156b/data/student_labels/train.csv"
	traindf = pd.read_csv(train)
	classesdf = traindf[classes][:-1]
	if fillna:
		classesdf = traindf[classes].fillna(naVal)[:-1]
	paths = traindf["Path"].tolist()[:-1]
	if partialData:
		classesdf = traindf[classes].iloc[:numdata]
		paths = traindf["Path"].iloc[:numdata].tolist()
	if processing == "simple":
		Xdf = np.array([preprocessing_simple(Image.open(prefix+path)) for path in paths])
	elif processing == "complex":
		Xdf = np.array([preprocessing_complex(Image.open(prefix+path)) for path in paths])
	else:
		Xdf = np.array([np.asarray(Image.open(prefix+path).resize((imagex, imagey))) for path in paths])
	return Xdf, classesdf

def get_dataLoader(Xdf, classesdf, classi, processing, ensemble=False, imagex=320, imagey=320):
	channels = 1
	if processing == "complex":
		channels = 3
	x_vals = Xdf
	y_vals = classesdf
	if ensemble:
		knownValues = ~classesdf[classi].isna()
		x_vals = Xdf[knownValues]
		y_vals = classesdf[classi].loc[knownValues]
	X_train = torch.from_numpy(x_vals.reshape((-1, channels, imagex, imagey)).astype('float32'))
	y_train = torch.from_numpy(y_vals.to_numpy().astype('float32'))
	train_dataset = TensorDataset(X_train, y_train)
	training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
	return training_data_loader

def load_testdata(partialData=False, numtest=10, imagex=320, imagey=320):
	prefix = "/groups/CS156b/data/"
	test = "/groups/CS156b/data/student_labels/test_ids.csv"
	testdf = pd.read_csv(test)
	testpaths = testdf["Path"].tolist()
	if partialData:
		testpaths = testdf["Path"].iloc[:numtest].tolist()
	if processing == "simple":
		Xtestdf = np.array([np.asarray(Image.open(prefix+path).resize((imagex, imagey))) for path in testpaths])
	elif processing == "complex":
		Xtestdf = np.array([np.asarray(Image.open(prefix+path).resize((imagex, imagey))) for path in testpaths])
	else:
		Xtestdf = np.array([np.asarray(Image.open(prefix+path).resize((imagex, imagey))) for path in testpaths])
	X_test = torch.from_numpy(Xtestdf.reshape((-1, 1, imagex, imagey)).astype('float32'))
	test_dataset = TensorDataset(X_test)
	test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
	ids = testdf['Id'].tolist()
	if partialData:
		ids = testdf['Id'].iloc[:numtest].tolist()
	return test_data_loader, ids

def get_CNN(device, processing, ensemble=False, updateWeights=False):
	channels = 1
	if processing == "complex":
		channels = 3
	out = 14
	if ensemble:
		out = 1
	model = nn.Sequential(
	    nn.Conv2d(channels, 64, kernel_size=(3,3)),
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
	    nn.Linear(64, out),
	    nn.Tanh()
	)
	return model

def get_densenet(device, processing, ensemble=False, updateWeights=False):
	channels = 1
	if processing == "complex":
		channels = 3
	out = 14
	if ensemble:
		out = 1
	model = models.densenet161(pretrained=True)
	model.features.conv0 = nn.Conv2d(channels, 96, kernel_size=7, stride=2, padding=3,bias=False)
	for param in model.parameters():
	    param.requires_grad = updateWeights
	model.classifier = nn.Sequential(nn.Linear(2208, 512),
	                                 nn.ReLU(),
	                                 nn.Dropout(0.2),
	                                 nn.Linear(512, out),
	                                 nn.LogSoftmax(dim=1),
	                                 nn.Tanh())
	return model

def get_inception(device, processing, ensemble=False, updateWeights=False):
	channels = 1
	if processing == "complex":
		channels = 3
	out = 14
	if ensemble:
		out = 1
	model = models.inception_v3(pretrained=True)
	model.transform_input = False
	model.Conv2d_1a_3x3 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=3, bias=False)
	model.fc = nn.Linear(2048, 1)
	return model

def get_resnet(device, processing, updateWeights=False):
	channels = 1
	if processing == "complex":
		channels = 3
	model = models.resnet50(pretrained=True)
	model.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3,bias=False)
	for param in model.parameters():
	    param.requires_grad = False
	model.fc = nn.Sequential(nn.Linear(2048, 512),
	                                 nn.ReLU(),
	                                 nn.Dropout(0.2),
	                                 nn.Linear(512, out),
	                                 nn.LogSoftmax(dim=1),
	                                 nn.Tanh())
	return model

def get_vgg(device, processing, ensemble=False, updateWeights=False):
	channels = 1
	if processing == "complex":
		channels = 3
	out = 14
	if ensemble:
		out = 1
	model = models.vgg16(pretrained=True)
	model.features[0] = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
	model.classifier[6] = nn.Linear(4096, out)
	return model

def fit_model(model, training_data_loader, device, n_epochs=20):
	criterion = nn.MSELoss()
	# criterion = nn.NLLLoss()
	# criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	model.to(device)
	training_loss_history = np.zeros([n_epochs, 1])
	for epoch in range(n_epochs):
	    print(f'Epoch {epoch+1}/{n_epochs}:', end='')
	    model.train()
	    for i, data in enumerate(training_data_loader):
	        images, labels = data
			labels = labels.type(torch.LongTensor)
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

def test_model(model, classes, test_data_loader, filename, ids):
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
	filename = "/home/kmcgraw/CS156b/predictions/densenet_MSE_none_1000.csv"
	batch_size = 64
	processing = "none" # none, simple, complex
	ensemble = False
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	Xdf, classesdf = load_traindata(processing, classes, partialData=True)
	test_data_loader, ids = load_testdata(processing)
	if ensemble:
		for i in range(len(classes)):
			model = get_densenet(device, processing)
			training_data_loader = get_dataLoader(Xdf, classesdf, classes[i], processing)
			trained_model = fit_model(model, training_data_loader, device)
			test_model(trained_model, classes, test_data_loader, filename, ids)
	else:
		model = get_densenet(device, processing)
		training_data_loader = get_dataLoader(Xdf, classesdf, 0, processing)
		trained_model = fit_model(model, training_data_loader, device)
		test_model(classes, test_data_loader, filename, ids)
