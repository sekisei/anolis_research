# coding: UTF-8
import numpy as np
import random

#データセットのロード
class dataset_loader():
    def __init__ (self, base_path = '', data_size = None):
        self.data_size = data_size
        self.input_data_shape = (self.data_size, 224, 224, 3)
        self.label_data_shape = (self.data_size)
        self.img_label_data_shape = (int(self.data_size / 2), 224, 224, 1)

        self.data_X = np.memmap(base_path + 'Data_X.dat', dtype = 'float16', mode = 'r', shape = self.input_data_shape)
        self.data_Y = np.memmap(base_path + 'Data_Y.dat', dtype = 'float16', mode = 'r', shape = self.label_data_shape)
        self.data_img_Y = np.memmap(base_path + 'Data_img_Y.dat', dtype = 'float16', mode = 'r', shape = self.img_label_data_shape)

        self.dataset = {'data_X': self.data_X, 'data_Y': self.data_Y, 'data_img_Y': self.data_img_Y}

        print('Input data shape => ', self.data_X.shape)
        print('Label data shape => ' , self.data_Y.shape)
        print('Image label data shape => ', self.data_img_Y.shape)

    #たまたまK=5で均等に分けられるデータセット
    def split_list_into_K_access_list(self, K = 5, access_list = None, shuffle = False):
        self.access_list = access_list[:]
        if shuffle == True: random.shuffle(self.access_list)
        self.K_hold_access_list = []
        self.each_size = int(len(self.access_list) / K)
        for self.k in range(0, K): self.K_hold_access_list.append(self.access_list[self.k*self.each_size : self.k*self.each_size + self.each_size])
        return self.K_hold_access_list

    def get_new_access_list(self, k_num = None, splitted_dataset = None, shuffle = False):
        self.splitted_dataset = splitted_dataset[:]
        self.train_access_list = []
        self.test_access_list = self.splitted_dataset[k_num]
        self.splitted_dataset.pop(k_num)
        for self.idx in range(0, len(self.splitted_dataset)): self.train_access_list.extend(self.splitted_dataset[self.idx])
        if shuffle == True: random.shuffle(self.train_access_list)
        return (self.train_access_list, self.test_access_list)

    #streamでデータの割合を決める
    def change_data_rate(self, stream = None, rate = None):
        if stream == 'S_cl' or stream == 'S_cl and S_self':
            return 1.0
        else:
            return rate

    def split_access_list_for_each_stream(self, stream = None, access_list = None, data_Y = None, rate = None):
        self.access_list = access_list[:]
        self.access_list_for_Se = []
        self.access_list_for_Scl_Sam = access_list[:]

        if stream == 'S_cl, S_self and S_ext':
            self.access_list_for_Se_size = int(len(self.access_list)*(1.0 - rate))
            self.access_list_for_Se_anolis = []
            self.access_list_for_Se_others = []
        
            self.label_val_0_cnt = 0
            self.label_val_1_cnt = 0

            #とりあえず各Stream用リスト作成
            #最終的にはSeの植物データ分リストは削除される
            for self.idx in self.access_list:
                if (data_Y[self.idx] == 0) and (self.label_val_0_cnt != self.access_list_for_Se_size):
                    self.label_val_0_cnt += 1
                    self.access_list_for_Se_others.extend([self.idx])
                elif (data_Y[self.idx] == 1) and (self.label_val_1_cnt != self.access_list_for_Se_size):
                    self.label_val_1_cnt += 1
                    self.access_list_for_Se_anolis.extend([self.idx])
        
            #各Stream用リストを使って元のリストの一部をremove
            for self.idx in self.access_list_for_Se_anolis: self.access_list.remove(self.idx)
            for self.idx in self.access_list_for_Se_others: self.access_list.remove(self.idx)

            self.access_list_for_Se = self.access_list_for_Se_anolis
            self.access_list_for_Scl_Sam = self.access_list

            print( '[Data size on stream] S_cl and S_self: ' + str(len(self.access_list_for_Scl_Sam)) + ', S_ext: ' + str(len(self.access_list_for_Se)))
    
            #SeはSclより扱うデータ数が少ないのでサイズ調整
            #多少サイズが違うが、Scl以上のサイズであれば問題なし
            if rate != 1.0:
                self.access_list_for_Se_resized = self.access_list_for_Se * (int(len(self.access_list_for_Scl_Sam) / len(self.access_list_for_Se)) + 1)
                self.access_list_for_Se = [self.access_list_for_Se_resized[self.idx_num] for self.idx_num, self.element in enumerate(self.access_list_for_Scl_Sam)]
        
        return self.access_list_for_Scl_Sam, self.access_list_for_Se

    
    '''
    #stream ext用のデータ再構築関数（リストから構築）
    def split_access_list_for_stream_ext(self, access_list = None, rate = None):
        #Se(stream ext)用のアクセスリストの切り取り方が固定だが、問題ない。（画像ラベルは決まったデータしか用意できないことが多いため）
        self.access_list_for_Scl = access_list[0:int(len(access_list)*rate)]
        self.access_list_for_Se = access_list[int(len(access_list)*rate):len(access_list)]
        print('[Data size on stream] S_cl, S_cl and S_self: ' + str(len(self.access_list_for_Scl)) + ', S_ext: ' + str(len(self.access_list_for_Se)))
        #SeはSclより扱うデータ数が少ないのでサイズ調整
        #多少サイズが違うが、Scl以上のサイズであれば問題なし
        if rate != 1.0:
            self.access_list_for_Se_resized = self.access_list_for_Se * (int(len(self.access_list_for_Scl) / len(self.access_list_for_Se)) + 1)
            self.access_list_for_Se = [self.access_list_for_Se_resized[self.idx_num] for self.idx_num, self.element in enumerate(self.access_list_for_Scl)]
        #print('list size: S_cl and S_self --> ' + str(len(self.access_list_for_Scl)) + ', S_ext -- > ' + str(len(self.access_list_for_Se)) + '(extentioned size)')
        return self.access_list_for_Scl, self.access_list_for_Se
    '''


    '''
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
    '''
        
if __name__ == '__main__':
    data_size = 20 #48710
    loader = dataset_loader(base_path = '/media/kai/4tb/anolis_dataset_for_DL/')
    splitted_dataset = loader.split_list_into_K_access_list(K = 5, data_size = data_size, shuffle = True)
    for k_num in range(0, 5):
        K_hold_access_list = loader.get_new_access_list(k_num = k_num, splitted_dataset = splitted_dataset)
        (train_data, test_data) = (K_hold_access_list[0], K_hold_access_list[1])
        print(train_data, test_data)
        print(loader.split_access_list_for_stream_ext(access_list = train_data, data_Y = np.array([1,0,1,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,0]), rate = 0.8))
        #print(len(train_data), len(test_data))
        #print(len(train_data)+len(test_data))
