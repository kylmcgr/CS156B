import numpy as np
import pandas as pd
from PIL import Image

prefix = "/groups/CS156b/data/"
train = "/groups/CS156b/data/student_labels/train.csv"
traindf = pd.read_csv(train)
paths = traindf["Path"].iloc[:1000].tolist()
Xdf = np.array([np.asarray(Image.open(prefix+path)) for path in paths], dtype=object)

np.savetxt('/home/bjuarez/CS156b/data/Xdf.csv', Xdf, delimiter=',', fmt='%s')