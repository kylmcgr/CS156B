import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
import cv2
from skimage import io, transform


prefix = "/groups/CS156b/data/"
train = "/groups/CS156b/data/student_labels/train.csv"

traindf = pd.read_csv(train)
paths = traindf["Path"].tolist()[:-1]

Xdf = np.array([np.asarray(Image.open(prefix+path).resize((256, 256))) for path in paths])
