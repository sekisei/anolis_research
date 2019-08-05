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

    #stream ext用のデータ再構築関数
    def make_access_list_for_stream_ext(self, data_list, percent = 0.5):
        def access_list_divider(access_list, per = 0.5):
            self.range0 = 0
            self.range1 = int(len(access_list) * per)
            self.range2 = len(access_list)
            self.access_list_for_first = access_list[self.range0 : self.range1]
            self.access_list_for_second = access_list[self.range1 : self.range2]
            #print('first data size: ' + str(len(self.access_list_for_first)) + ', second data size: ' + str(len(self.access_list_for_second)))
            return self.access_list_for_first, self.access_list_for_second

        self.anolis_only_list, self.others_only_list = access_list_divider(data_list, per = 0.5)
        self.anolis_list_for_Lself, self.anolis_list_for_Lext = access_list_divider(self.anolis_only_list, per = percent)
        self.others_list_for_Lself, self.others_list_for_Lext = access_list_divider(self.others_only_list, per = percent)
        self.anolis_list_for_Lself.extend(self.others_list_for_Lself)
        return self.anolis_list_for_Lself, self.anolis_list_for_Lext   

    def make_access_list(self, input_data, dataset_rate = 0.8, stream = 'S_cl'):
        self.access_list = list(range(0, len(input_data)))
        if stream == 'S_cl, S_self and S_ext':
           #gain->percent=0.8, no gain-> percent=1.0
           self.access_list_for_Scl, self.access_list_for_Se = self.make_access_list_for_stream_ext(self.access_list, percent = dataset_rate)
           #SeはSclより扱うデータ数が少ないのでサイズ調整（ランダムに構成）
           self.access_list_for_Se_resized = self.access_list_for_Se * (int(len(self.access_list_for_Scl) / len(self.access_list_for_Se)) + 1)
           self.access_list_for_Se = [self.access_list_for_Se_resized[self.idx_num] for self.idx_num, self.element in enumerate(self.access_list_for_Scl)]
           return self.access_list_for_Scl, self.access_list_for_Se
        else:
           return self.access_list 
        
