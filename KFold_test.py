#coding: UTF-8

import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import random

X = np.memmap('/media/kai/4tb/anolis_dataset_for_DL/data_X.dat', dtype = 'float16', mode = 'r', shape = (60000, 224, 224, 3))
Y = np.memmap('/media/kai/4tb/anolis_dataset_for_DL/data_Y.dat', dtype = 'float16', mode = 'r', shape = (60000, 1))#64840

#X = np.load('/media/kai/4tb/anolis_dataset_for_DL/X_test.npy')
#Y = np.load('/media/kai/4tb/anolis_dataset_for_DL/Y_test.npy')


kf = KFold(n_splits = 10, shuffle = False)

for train_index, test_index in kf.split(X):#tqdm(kf.split(X), total = kf.get_n_splits(X)):
    random.shuffle(train_index)
    print("TRAIN:", train_index, "TEST:", test_index)
    print(len(train_index), len(test_index))
    for batch in tqdm(range(0, 10)):
        X_train, X_test = X[train_index[batch]], X[test_index[batch]]
        Y_train, Y_test = Y[train_index[batch]], Y[test_index[batch]]


