import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import pickle

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

LABELED_PATH = '../data/labeled_img/'
TEST_PATH = '../data/test_img/'

IMAGE_SIZE = (256, 256)

# function to read image
def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, IMAGE_SIZE)
    return img

train['f_label'], labels = train['label'].factorize()
id_to_labels = {i: l for i, l in enumerate(labels.values)}

def read_train_data(d, prefix):
    X, Y = [], []
    for img_path, _, label in tqdm(d):
        X.append(read_img(prefix + img_path + '.png'))
        Y.append(label)
    return np.array(X, np.float32), np.array(Y)

x_train, y_train = read_train_data(train.values, LABELED_PATH)
x_train /= 255.

def read_test_data(d, prefix):
    ids = []
    X = []
    for (img_path,) in tqdm(d):
        ids.append(img_path)
        X.append(read_img(prefix + img_path + '.png'))
    return np.array(ids), np.array(X, np.float32)

ids, x_test = read_test_data(test.values, TEST_PATH)
x_test /= 255.

with open('../data/labeled_data.pickle', 'wb') as f:
    pickle.dump((x_train, y_train), f)

with open('../data/test_data.pickle', 'wb') as f:
    pickle.dump((ids, x_test), f)

with open('../data/id_to_labels.pickle', 'wb') as f:
    pickle.dump(id_to_labels, f)