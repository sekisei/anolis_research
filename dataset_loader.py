import numpy as np

#データセットのロード
class load():
    def __init__ (self, base_path = '/home'):
        self.test_num = 378
        self.valid_num = 5006
        self.train_num = 18971
    
        self.Label_test_shape = (self.test_num * 2, 224, 224, 1)
        self.Label_valid_shape = (self.valid_num * 2, 224, 224, 1)
        self.Label_train_shape = (self.train_num * 2, 224, 224, 1)
    
        self.Input_test_shape = (self.test_num * 2, 224, 224, 3)
        self.Input_valid_shape = (self.valid_num * 2, 224, 224, 3)
        self.Input_train_shape = (self.train_num * 2, 224, 224, 3)
    
        self.test_X = np.memmap(base_path + 'test_X.dat', dtype = 'float16', mode = 'r', shape = self.Input_test_shape)
        self.train_X = np.memmap(base_path + 'train_X.dat', dtype = 'float16', mode = 'r', shape = self.Input_train_shape)
        self.valid_X = np.memmap(base_path + 'valid_X.dat', dtype = 'float16', mode = 'r', shape = self.Input_valid_shape)
        self.test_Y = np.memmap(base_path + 'test_Y.dat', dtype = 'float16', mode = 'r', shape = (self.test_num * 2))
        self.test_img_Y = np.memmap(base_path + 'test_img_Y.dat', dtype = 'float16', mode = 'r', shape = self.Label_test_shape)
        self.train_Y = np.memmap(base_path + 'train_Y.dat', dtype = 'float16', mode = 'r', shape = (self.train_num * 2))
        self.train_img_Y = np.memmap(base_path + 'train_img_Y.dat', dtype = 'float16', mode = 'r', shape = self.Label_train_shape)
        self.valid_Y = np.memmap(base_path + 'valid_Y.dat', dtype = 'float16', mode = 'r', shape = (self.valid_num * 2))
        self.valid_img_Y = np.memmap(base_path + 'valid_img_Y.dat', dtype = 'float16', mode = 'r', shape = self.Label_valid_shape)

        print(len(self.test_X))
        print(len(self.train_X))
        print(len(self.valid_X)) 
