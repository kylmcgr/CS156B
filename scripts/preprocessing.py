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
import sys


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

def load_traindata(processing, classes, partialData=False, numdata=1000, imagex=320, imagey=320):
	prefix = "/groups/CS156b/data/"
	train = "/groups/CS156b/data/student_labels/train.csv"
	traindf = pd.read_csv(train)
	# classesdf = traindf[classes].fillna(0)[:-1]
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

def get_dataLoader(Xdf, classesdf, classi, processing, imagex=320, imagey=320):
	channels = 1
	if processing == "complex":
		channels = 3
	knownValues = ~classesdf[classi].isna()
	x_vals = Xdf[knownValues]
	y_vals = classesdf[classi].loc[knownValues]
	X_train = torch.from_numpy(x_vals.reshape((-1, channels, imagex, imagey)).astype('float32'))
	y_train = torch.from_numpy(y_vals.to_numpy().astype('float32'))
	train_dataset = TensorDataset(X_train, y_train)
	training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
	return training_data_loader

def load_testdata(processing, partialData=False, numtest=10, imagex=320, imagey=320):
	channels = 1
	if processing == "complex":
		channels = 3
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
	X_test = torch.from_numpy(Xtestdf.reshape((-1, channels, imagex, imagey)).astype('float32'))
	test_dataset = TensorDataset(X_test)
	test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
	ids = testdf['Id'].tolist()
	if partialData:
		ids = testdf['Id'].iloc[:numtest].tolist()
	return test_data_loader, ids

if __name__ == "__main__":
    classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices']
    batch_size = 64
    imagex, imagey = 320, 320
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Xdf, classesdf = load_traindata(processing, classes, partialData=True, numdata=10, imagex=imagex, imagey=imagey)
	X_train = torch.from_numpy(Xdf.reshape((-1, channels, imagex, imagey)).astype('float32'))
	y_train = torch.from_numpy(classesdf.to_numpy().astype('float32'))
	train_dataset = TensorDataset(X_train, y_train)
	training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_data_loader, ids = load_testdata(processing, partialData=True, numtest=10)
