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
import cv2
import skimage.io
import skimage.color
import skimage.filters
from sklearn import preprocessing

partial_data = False
pp = True
pp_complex = False
fill = -1
frozen = True
final_layer_complex = False
tanh = True
resizex = 320
resizey = 320

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

classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices']

batch_size = 64
n_epochs = 20

traindf = pd.read_csv("/groups/CS156b/data/student_labels/train.csv")

if partial_data:
    classesdf = traindf[classes].fillna(fill).iloc[:1000]
    paths = traindf["Path"].iloc[:1000].tolist()
else:
    classesdf = traindf[classes].fillna(fill) 
    paths = traindf["Path"].tolist()[:-1]

if pp:
    if pp_complex:
        Xdf = np.array([preprocessing_complex(Image.open("/groups/CS156b/data/"+path)) for path in paths])
    else:
        Xdf = np.array([preprocessing_simple(Image.open("/groups/CS156b/data/"+path)) for path in paths])
else:
    Xdf = np.array([np.asarray(Image.open("/groups/CS156b/data/"+path).resize((resizex, resizey))) for path in paths])

X_train = torch.from_numpy(Xdf.reshape((-1, 1, resizex, resizey)).astype('float32'))

if partial_data:
    y_train = torch.from_numpy((classesdf+1).to_numpy().astype('float32'))
else:
    y_train = torch.from_numpy((classesdf+1).to_numpy().astype('float32')[:-1]) 
    
train_dataset = TensorDataset(X_train, y_train)
training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda:0")

model = models.inception_v3(pretrained=True)

model.transform_input = False

num_channels = 1
if pp_complex:
    num_channels = 3
model.Conv2d_1a_3x3 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=3, bias=False)

if frozen:
    for param in model.parameters():
        param.requires_grad = False

if final_layer_complex:
    model.fc = nn.Sequential(nn.Linear(2048, 512), 
                                nn.ReLU(), 
                                nn.Dropout(0.2), 
                                nn.Linear(512, 14), 
                                nn.LogSoftmax(dim=1), 
                                nn.Tanh())
elif tanh:
    model.fc = nn.Sequential(nn.Linear(2048, 14),
                                nn.Tanh())
else:
    model.fc = nn.Linear(2048, 14)
    
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
    training_loss_history[epoch] /= len(training_data_loader)
    print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}',end='')
    
testdf = pd.read_csv("/groups/CS156b/data/student_labels/test_ids.csv")

if partial_data:
    testpaths = testdf["Path"].iloc[:10].tolist()
else:
    testpaths = testdf["Path"].tolist()

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
if partial_data:
    outdf.insert(0, 'Id', testdf['Id'].iloc[:10].tolist())
else:
    outdf.insert(0, 'Id', testdf['Id'].tolist())
outdf.to_csv("/home/bjuarez/CS156b/predictions/inceptionV3_fill-1_tanh_frozen_pp=simple_data=full.csv", index=False)

