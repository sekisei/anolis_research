import numpy as np
from tqdm import tqdm

test_num = 378
valid_num = 5006
train_num = 18971

others_test_class_Label = np.array([[0, 1] for index in range(0, test_num)])
anolis_test_class_Label = np.array([[1, 0] for index in range(0, test_num)])
others_train_class_Label = np.array([[0, 1] for index in range(0, train_num)])
anolis_train_class_Label = np.array([[1, 0] for index in range(0, train_num)])
others_valid_class_Label = np.array([[0, 1] for index in range(0, valid_num)])
anolis_valid_class_Label = np.array([[1, 0] for index in range(0, valid_num)])

test_label_stacked = np.vstack((anolis_test_class_Label, others_test_class_Label))
#print(test_label_stacked.shape)
test = np.memmap('test_Y.dat', dtype = 'float16', mode = 'w+', shape = test_label_stacked.shape)
test[:] = test_label_stacked[:]
test.flush()

train_label_stacked = np.vstack((anolis_train_class_Label, others_train_class_Label))
train = np.memmap('train_Y.dat', dtype = 'float16', mode = 'w+', shape = train_label_stacked.shape)
train[:] = train_label_stacked[:]
train.flush()

valid_label_stacked = np.vstack((anolis_valid_class_Label, others_valid_class_Label))
valid = np.memmap('valid_Y.dat', dtype = 'float16', mode = 'w+', shape = valid_label_stacked.shape)
valid[:] = valid_label_stacked[:]
valid.flush()
