import numpy as np
import sys

args = sys.argv

print('PATH(load), PATH(save)')
base_path_to_load = args[1]
base_path_to_save = args[2]

test_num = 378
valid_num = 5006
train_num = 18971

Label_test_shape = (test_num * 2, 224, 224, 1)
Label_valid_shape = (valid_num * 2, 224, 224, 1)
Label_train_shape = (train_num * 2, 224, 224, 1)

Input_test_shape = (test_num * 2, 224, 224, 3)
Input_valid_shape = (valid_num * 2, 224, 224, 3)
Input_train_shape = (train_num * 2, 224, 224, 3)
        
test_X = np.memmap(base_path_to_load + 'test_X.dat', dtype = 'float16', mode = 'r', shape = Input_test_shape)
train_X = np.memmap(base_path_to_load + 'train_X.dat', dtype = 'float16', mode = 'r', shape = Input_train_shape)
valid_X = np.memmap(base_path_to_load + 'valid_X.dat', dtype = 'float16', mode = 'r', shape = Input_valid_shape)
test_Y = np.memmap(base_path_to_load + 'test_Y.dat', dtype = 'float16', mode = 'r', shape = (test_num * 2))
test_img_Y = np.memmap(base_path_to_load + 'test_img_Y.dat', dtype = 'float16', mode = 'r', shape = Label_test_shape)
train_Y = np.memmap(base_path_to_load + 'train_Y.dat', dtype = 'float16', mode = 'r', shape = (train_num * 2))
train_img_Y = np.memmap(base_path_to_load + 'train_img_Y.dat', dtype = 'float16', mode = 'r', shape = Label_train_shape)
valid_Y = np.memmap(base_path_to_load + 'valid_Y.dat', dtype = 'float16', mode = 'r', shape = (valid_num * 2))
valid_img_Y = np.memmap(base_path_to_load + 'valid_img_Y.dat', dtype = 'float16', mode = 'r', shape = Label_valid_shape)

print(len(test_X))
print(len(train_X))
print(len(valid_X))

def stack_and_save(np_array1, np_array2, np_array3, file_name, mode = 'v'):
    if mode == 'v':
        stacked = np.vstack((np_array1, np_array2))
        stacked = np.vstack((stacked, np_array3))
    else:
        stacked = np.hstack((np_array1, np_array2))
        stacked = np.hstack((stacked, np_array3))
        
    mem = np.memmap(base_path_to_save + file_name, dtype = 'float16', mode = 'w+', shape = stacked.shape)
    mem[:] = stacked[:]
    mem.flush()
    print(stacked.shape)
    print(file_name + ' <- done')

stack_and_save(train_X, valid_X, test_X, 'Data_X.dat', mode = 'v')
stack_and_save(train_Y, valid_Y, test_Y, 'Data_Y.dat', mode = 'h')
stack_and_save(train_img_Y, valid_img_Y, test_img_Y, 'Data_img_Y.dat', mode = 'v')
