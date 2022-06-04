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
	bilateral = cv2.bilateralFilter(new_image, 5, 50, 50)
	block_size =75
	local_thresh = skimage.filters.threshold_local(new_image, block_size, offset=5)
	binary_local = new_image > local_thresh
	gaussHist = skimage.exposure.equalize_hist(new_image)
	max = 0
	min = new_image[0,0]
	for i in range(new_image.shape[0]):
	    for j in range(new_image.shape[1]):
	        if new_image[i,j] > max:
	            max = new_image[i,j]
	        if new_image[i,j] < min:
	            min = new_image[i,j]
	t = min + 0.9 * (max-min)
	binary_mask = new_image < t
	area_closed = skimage.morphology.area_closing(binary_mask,area_threshold = 128)
	total_img = np.stack([gaussHist,binary_local,bilateral],axis=-1)
	selection = total_img.copy()
	selection[~area_closed] = 0
	return selection
	
def preprocessing_simple(image):
	new_image = image.resize((320, 320))
	new_image = np.float32(new_image)
	bilateral = cv2.bilateralFilter(new_image, 5, 50, 50)
	blurred_image = skimage.filters.gaussian(bilateral, sigma=1.0)
	max = 0
	min = blurred_image[0,0]
	for i in range(blurred_image.shape[0]):
	    for j in range(blurred_image.shape[1]):
	        if blurred_image[i,j] > max:
	            max = blurred_image[i,j]
	        if blurred_image[i,j] < min:
	            min = blurred_image[i,j]
	t = min + 0.9 * (max-min)
	binary_mask = blurred_image < t
	area_closed = skimage.morphology.area_closing(binary_mask,area_threshold = 128)
	selection = blurred_image.copy()
	selection[~area_closed] = 0
	return selection
	
def load_traindata(processing, classes, partial_data, naVal, fillna=True, numdata=1000, resizex=320, resizey=320):
    traindf = pd.read_csv("/groups/CS156b/data/student_labels/train.csv")
    # classesdf = traindf["Path"][:-1] # potential issue
    # if fillna:
    #     classesdf = traindf[classes].fillna(naVal)[:-1] # potential issue 
    classesdf = traindf[classes].fillna(naVal) # [:-1]
    paths = traindf["Path"].tolist()[:-1]    
    if partial_data:
        classesdf = traindf[classes].fillna(naVal).iloc[:numdata]
        # if fillna:
	       # classesdf = traindf[classes].fillna(naVal).iloc[:numdata]
        paths = traindf["Path"].iloc[:numdata].tolist()
    if processing == "simple":
        Xdf = np.array([preprocessing_simple(Image.open("/groups/CS156b/data/"+path)) for path in paths])
    elif processing == "complex":
        Xdf = np.array([preprocessing_complex(Image.open("/groups/CS156b/data/"+path)) for path in paths])
    else:
        Xdf = np.array([np.asarray(Image.open("/groups/CS156b/data/"+path).resize((resizex, resizey))) for path in paths])
    return Xdf, classesdf
    
def get_dataLoader(Xdf, classesdf, processing, partial_data, resizex=320, resizey=320):
    num_channels = 1
    if processing == "complex":
        num_channels = 3
    X_train = torch.from_numpy(Xdf.reshape((-1, num_channels, resizex, resizey)).astype('float32'))
    # been using "classesdf+1" which rescales...making note of this
    y_train = torch.from_numpy((classesdf).to_numpy().astype('float32')[:-1]) # potential issue: classesdf or classesdf+1
    if partial_data:
        y_train = torch.from_numpy((classesdf).to_numpy().astype('float32'))
    train_dataset = TensorDataset(X_train, y_train)
    training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return training_data_loader
    
def load_testdata(processing, partial_data, resizex=320, resizey=320, numtest=10):
    testdf = pd.read_csv("/groups/CS156b/data/student_labels/test_ids.csv")
    testpaths = testdf["Path"].tolist()
    if partial_data:
        testpaths = testdf["Path"].iloc[:numtest].tolist()
    if processing == "simple":
        Xtestdf = np.array([preprocessing_simple(Image.open("/groups/CS156b/data/"+path)) for path in testpaths])
        X_test = torch.from_numpy(Xtestdf.reshape((-1, 1, resizex, resizey)).astype('float32'))
    elif processing == "complex":
        Xtestdf = np.array([preprocessing_complex(Image.open("/groups/CS156b/data/"+path)) for path in testpaths])
        X_test = torch.from_numpy(Xtestdf.reshape((-1, 3, resizex, resizey)).astype('float32'))
    else:
        Xtestdf = np.array([np.asarray(Image.open("/groups/CS156b/data/"+path).resize((resizex, resizey))) for path in testpaths])
        X_test = torch.from_numpy(Xtestdf.reshape((-1, 1, resizex, resizey)).astype('float32'))
    test_dataset = TensorDataset(X_test)
    test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    ids = testdf["Id"].tolist()
    if partial_data:
        ids = testdf["Id"].iloc[:numtest].tolist()
    return test_data_loader, ids

def get_model(device, processing, updateWeights, withTanh):
    num_channels = 1
    if processing == "complex":
        num_channels = 3
    model = models.inception_v3(pretrained=True)
    model.transform_input = False
    model.Conv2d_1a_3x3 = nn.Conv2d(num_channels, 32, kernel_size=3, stride=2, padding=3, bias=False)
    for param in model.parameters():
        param.requires_grad = updateWeights
    model.fc = nn.Linear(2048, 14)
    if withTanh:
        model.fc = nn.Sequential(nn.Linear(2048, 14),
                                nn.Tanh())
    return model
    
def fit_model(model, training_data_loader, device, n_epochs=20):
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
            loss = criterion(output.logits, labels) # check logits
            loss.backward()
            optimizer.step()
            training_loss_history[epoch] += loss.item()
            if i % 180 == 0: 
                print('.',end='')
        training_loss_history[epoch] /= len(training_data_loader)
        print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}',end='')
    print('\n\n')
    return model
    
def test_model(model, classes, test_data_loader, filename, ids):
    out = np.empty((0,len(classes)), int)
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(test_data_loader):
            images = data[0].to(device)
            output = model(images).cpu().numpy()
            out = np.append(out, output, axis=0)
    outdf = pd.DataFrame(data = out, columns=classes) # potential error
    outdf.insert(0, 'Id', ids)
    outdf.to_csv(filename, index=False)
	
if __name__ == "__main__":
    if len(sys.argv) < 7:
        print("Invalid number of arguments")
        sys.exit()
    processing = sys.argv[1] # processing = noPP, simple, or complex
    if sys.argv[2] == "frozen":
        updateWeights = False
    else:
        updateWeights = True
    if sys.argv[3] == "tanh":
        withTanh = True
    else:
        withTanh = False
    if sys.argv[4] == "-1":
        naVal = -1
    elif sys.argv[4] == "0":
        naVal = 0
    elif sys.argv[4] == "-0.5":
        naVal = -0.5
    else:
        print("Invalid naVal argument")
        sys.exit()
    if sys.argv[5] == "partial":
        partial_data = True
    else:
        partial_data = False
    if sys.argv[6] == "use_PP_data":
        use_PP_data = True
    else:
        use_PP_data = False
    
    classes = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
            'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices']
    filename = "/home/bjuarez/CS156b/predictions/inception_" + processing + "_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + ".csv"
    batch_size = 64
    
    device = torch.device("cuda:0")
    
    if use_PP_data:
        #print("bing")
        traindf = pd.read_csv("/groups/CS156b/data/student_labels/train.csv")
        classesdf = traindf[classes].fillna(naVal).iloc[0:120000]
        y_train = torch.from_numpy((classesdf).to_numpy().astype('float32')) # [:-1]
        #print("\nbeep")
        t0 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=0_size=15000.pt")
        #print("\nboop")
        t1 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=1_size=15000.pt")
        t2 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=2_size=15000.pt")
        t3 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=3_size=15000.pt")
        t4 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=4_size=15000.pt")
        t5 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=5_size=15000.pt")
        t6 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=6_size=15000.pt")
        t7 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=7_size=15000.pt")
        # t8 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=8_size=15000.pt")
        # t9 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=9_size=15000.pt")
        # t10 = torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_train_data_simple_naVal=0_split=10_size=15000.pt")
        X_train_PP = torch.cat((t0,t1,t2,t3,t4,t5,t6,t7), dim=0)
        
        train_dataset = TensorDataset(X_train_PP, y_train)
        training_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        testdf = pd.read_csv("/groups/CS156b/data/student_labels/solution_ids.csv")
        test_dataset = TensorDataset(torch.load("/groups/CS156b/2022/team_dirs/DJJ/processed_solution_data_simple.pt")) # wait for solution data
        test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        ids = testdf["Id"].tolist()
        
        model = get_model(device, processing, updateWeights, withTanh)
        trained_model = fit_model(model, training_data_loader, device)
        filename = "/home/bjuarez/CS156b/predictions/SOL_inception_t0-7_with_PP_DATA_" + sys.argv[2] + "_" + sys.argv[3] + "_" + sys.argv[4] + "_" + sys.argv[5] + ".csv"
        test_model(trained_model, classes, test_data_loader, filename, ids)
    else:
        Xdf, classesdf = load_traindata(processing, classes, partial_data, naVal)
        test_data_loader, ids = load_testdata(processing, partial_data)
        model = get_model(device, processing, updateWeights, withTanh)
        training_data_loader = get_dataLoader(Xdf, classesdf, processing, partial_data)
        trained_model = fit_model(model, training_data_loader, device)
        test_model(trained_model, classes, test_data_loader, filename, ids)
    
    