# coding: UTF-8
import numpy as np
import random
import math

#データセットのロード
class dataset_loader():
    def __init__ (self, path = ''):
        self.path = path
        self.all_data_size = 64840

        self.mem_X = np.memmap(path + 'data_X.dat', dtype = 'float16', mode = 'r', shape = (self.all_data_size, 224, 224, 3))
        self.mem_Y = np.memmap(path + 'data_Y.dat', dtype = 'float16', mode = 'r', shape = (self.all_data_size, 1))
        self.mem_Y_img = np.memmap(path + 'data_Y_img.dat', dtype = 'float16', mode = 'r', shape = (self.all_data_size, 224, 224, 1))

        self.dataset = {'X': self.mem_X, 'Y': self.mem_Y, 'Y_img': self.mem_Y_img}

        print('Y_img: Input data shape => {Y_img_shape}'.format(Y_img_shape = self.mem_Y_img.shape))
        print('    X: Input data shape => {X_shape}'.format(X_shape = self.mem_X.shape))
        print('    Y: Input data shape => {Y_shape}'.format(Y_shape = self.mem_Y.shape))
    
    def expand_index(self, index = None, size = None):
        if len(index) == 0:
            print('Index_size is ZERO (No index expanding)')
            return index
        self.size_rate = math.ceil(float(size) / float(len(index)))
        self.idx_expanded = index * self.size_rate
        return self.idx_expanded

    def split_access_list_for_each_stream(self, stream = None, train_idx = None, default_rate = None):
        self.rate = default_rate
        if stream == 'S_cl' or stream == 'S_cl and S_self': self.rate = (1.0, 0)
        self.Scl_Sam_idx_size, self.Se_idx_size = int(self.rate[0] * len(train_idx)), int(self.rate[1] * len(train_idx))
        self.Scl_Sam_idx = train_idx[0 : self.Scl_Sam_idx_size] 
        self.Se_idx = train_idx[self.Scl_Sam_idx_size : self.Scl_Sam_idx_size + self.Se_idx_size]
        print('[Index size] Scl Sam: {Scl_Sam_idx_size}, Se: {Se_idx_size}'.format(Scl_Sam_idx_size = self.Scl_Sam_idx_size, Se_idx_size = self.Se_idx_size))
        return self.Scl_Sam_idx, self.Se_idx
        
    def load_dataset(self): return self.dataset    

if __name__ == '__main__':
    loader = dataset_loader(path = '/media/kai/4tb/anolis_dataset_for_DL/')
