# coding: UTF-8
# --memo--
#access_list ではなく idx_listのほうが適切な表現なので今後変更予定
import math
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
import random
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
import cv2

from tqdm import tqdm
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.losses import binary_crossentropy

#my module
import common_tools
import tensorflow_computer_vision_tools
import gain
    
class Set():
    def __init__(self, sess = None, stream = 'S_cl', learning_rate = 1.0e-8): #1.0e-8 
        
        self.stream = stream
        self.my_common_tools = common_tools.tools()
        self.my_cv_tools = tensorflow_computer_vision_tools.computer_vision_tools()

        self.gain = gain.load()
        self.Loss_of_stream = self.gain.set_stream(stream = self.stream)
        self.target = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.Loss_of_stream)

        #予測結果・誤差集計用リスト
        #self.acc_list_on_epoch = {'train': [], 'valid': [], 'test': []}
        #self.loss_list_on_epoch = {'train': [], 'valid': [], 'test': []}
    
        self.sess = K.get_session()
        self.uninitialized_variables = [self.v for self.v in tf.global_variables() if not hasattr(self.v, '_keras_initialized') or not self.v._keras_initialized]
        #print(sess.run(tf.report_uninitialized_variables(tf.global_variables())))
        self.sess.run(tf.variables_initializer(self.uninitialized_variables))
        self.sess.run(tf.initializers.local_variables())

        self.saver = tf.train.Saver(max_to_keep = None)

    #Scl, Sself, Sextで学習方法を分割
    def Train(self, epochs = 50, batch_size = 1, save_weights = True, save_default_weights = True, stream = None, access_list = None, input_data = None):
        print('[Training]: stream ({stm})'.format(stm = stream))
        (self.X, self.Y, self.Y_img) = (input_data[0], input_data[1], input_data[2])
        (self.access_list_for_Scl_origin, self.access_list_for_Se_origin) = (access_list[0], access_list[1])

        self.acc_list_on_epoch = [0 for self.e in range(0, epochs)]
        self.IoU_list_on_epoch = [0 for self.e in range(0, epochs)]
        self.loss_list_on_epoch = [0 for self.e in range(0, epochs)]

        self.path_to_default_weights = '/media/kai/4tb/ckpt_data/default_weights/my_model.ckpt'
        self.path_to_weights = lambda epoch: '/media/kai/4tb/ckpt_data/'+ str(epoch) +'/my_model' + str(epoch) + '.ckpt'
        
        if save_default_weights == True: self.saver.save(self.sess, self.path_to_default_weights)

        if stream == 'S_cl':
            for self.epoch in range(0, epochs):
                print('[epoch]: ' + str(self.epoch))

                self.c_sum_list_on_batch = []
                self.loss_sum_list_on_batch = []

                #Attention Mapの保存
                self.img_num = 22
                self.INPUT = self.X[self.img_num : self.img_num + 1]
                self.Attention_Map_val = self.gain.resized_AM.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.INPUT}) 
                self.HeatMap_val = self.my_cv_tools.get_Heat_Map(self.Attention_Map_val).reshape((224, 224, 3)) 
                self.my_cv_tools.save_AM(Heat_Map = self.HeatMap_val, target = self.INPUT, name_numbering = self.epoch)
                save_img('Input'+str(self.epoch)+'.png', self.INPUT.reshape((224, 224, 3))) 
                
                if save_weights == True: self.saver.save(self.sess, self.path_to_weights(self.epoch))
                self.access_list_for_Scl  = [idx for idx in self.access_list_for_Scl_origin]
                random.shuffle(self.access_list_for_Scl)
                
                while self.access_list_for_Scl != []:
                    self.idx = self.access_list_for_Scl[0 : batch_size]
                    self.X_batch = self.X[self.idx]
                    self.Y_batch = self.Y[self.idx]

                    print('{rem} remaining.'.format(rem = len(self.access_list_for_Scl)), end = '\r')
                    
                    [self.min_res, self.c_sum, self.l_sum] = self.sess.run(
                        [self.target, self.gain.correct_sum, self.gain.loss_sum_Lcl],
                        feed_dict = {self.gain.x_cl: self.X_batch, self.gain.t: self.Y_batch}
                    )
                    del self.access_list_for_Scl[0 : batch_size]
                    
                    self.c_sum_list_on_batch.append(self.c_sum)
                    self.loss_sum_list_on_batch.append(self.l_sum)
                
                self.acc_list_on_epoch[self.epoch] = sum(self.c_sum_list_on_batch) / len(self.access_list_for_Scl_origin)
                self.loss_list_on_epoch[self.epoch] = sum(self.loss_sum_list_on_batch) / len(self.access_list_for_Scl_origin)
                print('[Hist]: (Acc): {Acc}, (Loss): {Loss}'.format(Acc = self.acc_list_on_epoch[self.epoch], Loss = self.loss_list_on_epoch[self.epoch]))

        if stream == 'S_cl and S_am':
            for self.epoch in range(0, epochs):
                print('[epoch]: ' + str(self.epoch))

                self.c_sum_list_on_batch = []
                self.loss_sum_list_on_batch = []

                #Attention Mapの保存
                self.img_num = 22
                self.INPUT = self.X[self.img_num : self.img_num + 1]
                self.Attention_Map_val = self.gain.resized_AM.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.INPUT}) 
                self.HeatMap_val = self.my_cv_tools.get_Heat_Map(self.Attention_Map_val).reshape((224, 224, 3)) 
                self.my_cv_tools.save_AM(Heat_Map = self.HeatMap_val, target = self.INPUT, name_numbering = self.epoch)
                save_img('Input'+str(self.epoch)+'.png', self.INPUT.reshape((224, 224, 3))) 
                
                if save_weights == True: self.saver.save(self.sess, self.path_to_weights(self.epoch))
                self.access_list_for_Scl  = [idx for idx in self.access_list_for_Scl_origin]
                random.shuffle(self.access_list_for_Scl)
                
                while self.access_list_for_Scl != []:
                    self.idx = self.access_list_for_Scl[0 : batch_size]
                    self.X_batch = self.X[self.idx]
                    self.Y_batch = self.Y[self.idx]
                    self.Masked_val = self.gain.Masked_img.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.X_batch})

                    print('{rem} remaining.'.format(rem = len(self.access_list_for_Scl)), end = '\r')

                    [self.min_res, self.c_sum, self.l_sum] = self.sess.run(
                        [self.target, self.gain.correct_sum, self.gain.loss_sum_Lself],
                        feed_dict = {self.gain.x_cl: self.X_batch, self.gain.x_masked: self.Masked_val, self.gain.t: self.Y_batch}
                    )
                    del self.access_list_for_Scl[0 : batch_size]
                    
                    self.c_sum_list_on_batch.append(self.c_sum)
                    self.loss_sum_list_on_batch.append(self.l_sum)
                    
                self.acc_list_on_epoch[self.epoch] = sum(self.c_sum_list_on_batch) / len(self.access_list_for_Scl_origin)
                self.loss_list_on_epoch[self.epoch] = sum(self.loss_sum_list_on_batch) / len(self.access_list_for_Scl_origin)
                print('[Hist]: (Acc): {Acc}, (Loss): {Loss}'.format(Acc = self.acc_list_on_epoch[self.epoch], Loss = self.loss_list_on_epoch[self.epoch]))

        if stream == 'S_cl, S_am and S_e':
            for self.epoch in range(0, epochs):
                print('[epoch]: ' + str(self.epoch))

                self.c_sum_list_on_batch = []
                self.loss_sum_list_on_batch = []
                
                #Attention Mapの保存
                self.img_num = 22
                self.INPUT = self.X[self.img_num : self.img_num + 1]
                self.Attention_Map_val = self.gain.resized_AM.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.INPUT}) 
                self.HeatMap_val = self.my_cv_tools.get_Heat_Map(self.Attention_Map_val).reshape((224, 224, 3)) 
                self.my_cv_tools.save_AM(Heat_Map = self.HeatMap_val, target = self.INPUT, name_numbering = self.epoch)
                save_img('Input'+str(self.epoch)+'.png', self.INPUT.reshape((224, 224, 3))) 
                
                if save_weights == True: self.saver.save(self.sess, self.path_to_weights(self.epoch))

                (self.access_list_for_Scl, self.access_list_for_Se) = ([idx for idx in self.access_list_for_Scl_origin], [idx for idx in self.access_list_for_Se_origin])
                random.shuffle(self.access_list_for_Scl)
                #random.shuffle(self.access_list_for_Se) # <- shuffleはしないこと（Scl, Sam用のリストより大きめなので）

                while self.access_list_for_Scl != []:
                    self.idx = self.access_list_for_Scl[0 : batch_size]
                    self.idx_Se = self.access_list_for_Se[0 : batch_size]
                    self.X_batch = self.X[self.idx]
                    self.Y_batch = self.Y[self.idx]
                    self.X_Se_batch = self.X[self.idx_Se]
                    self.Y_Se_batch = self.Y_img[self.idx_Se]
                    self.Masked_val = self.gain.Masked_img.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.X_batch})

                    print('[Scl Sam]: {rem}, [Se]: {rem2} remaining.'.format(rem = len(self.access_list_for_Scl), rem2 = len(self.access_list_for_Se)), end = '\r')

                    self.feed_dict = {
                        self.gain.x_cl: self.X_batch,
                        self.gain.x_masked: self.Masked_val,
                        self.gain.x_ext: self.X_Se_batch,
                        self.gain.t: self.Y_batch,
                        self.gain.t_img: self.Y_Se_batch
                    }
                    
                    [self.min_res, self.c_sum, self.l_sum] = self.sess.run(
                        [self.target, self.gain.correct_sum, self.gain.loss_sum_Lext],
                        feed_dict = self.feed_dict
                    )

                    del self.access_list_for_Scl[0 : batch_size]
                    del self.access_list_for_Se[0 : batch_size]
                    
                    self.c_sum_list_on_batch.append(self.c_sum)
                    self.loss_sum_list_on_batch.append(self.l_sum)
                    
                self.acc_list_on_epoch[self.epoch] = sum(self.c_sum_list_on_batch) / len(self.access_list_for_Scl_origin)
                self.loss_list_on_epoch[self.epoch] = sum(self.loss_sum_list_on_batch) / len(self.access_list_for_Scl_origin)

                print(sum(self.c_sum_list_on_batch))
                print(len(self.access_list_for_Scl))

                print(
                    '[Hist]: (Acc): {Acc}, (Loss): {Loss}'.format(
                        Acc = self.acc_list_on_epoch[self.epoch],
                        Loss = self.loss_list_on_epoch[self.epoch]
                    )
                )
                
        print(self.acc_list_on_epoch)
        print(self.loss_list_on_epoch)

        return {'acc': self.acc_list_on_epoch, 'loss': self.loss_list_on_epoch}
    
    def Test(self, batch_size = 1, idx_list_for_test = None, input_data = None):
        print('Testing...')
        
        self.data_X, self.data_Y, self.data_img_Y = input_data['X'], input_data['Y'], input_data['Y_img']
        self.c_sum_list_on_batch = []
        self.loss_list_on_batch = []
        self.IoU_list_on_batch = []

        self.idx_list_origin = idx_list_for_test[0]
        self.idx_list_positive_only_origin = idx_list_for_test[1]
        self.idx_list = [self.idx for self.idx in self.idx_list_origin]
        self.idx_list_positive_only = [self.idx for self.idx in self.idx_list_positive_only_origin]

        print('[Size]: -Classification-: {size}'.format(size = len(self.idx_list_origin)))
        while self.idx_list != []:
            self.idx = self.idx_list[0 : batch_size]
            self.test_X_batch = self.data_X[self.idx]
            self.test_Y_batch = self.data_Y[self.idx]
            
            [self.c_sum, self.l_sum] = self.sess.run(
                [self.gain.correct_sum, self.gain.loss_sum_Lcl],
                feed_dict = {self.gain.x_cl: self.test_X_batch, self.gain.t: self.test_Y_batch, self.gain.dropout_mode: False}
            )
            
            del self.idx_list[0 : batch_size]

            self.c_sum_list_on_batch.append(self.c_sum)
            self.loss_list_on_batch.append(self.l_sum)

        print('[Size]: -IoU-: {size}'.format(size = len(self.idx_list_positive_only_origin)))
        while self.idx_list_positive_only != []:
            self.idx = self.idx_list_positive_only[0 : batch_size]
            self.test_X_batch = self.data_X[self.idx]
            self.test_img_Y_batch = self.data_img_Y[self.idx]

            [self.IoU] = self.sess.run(
                [self.gain.IoU],
                feed_dict = {self.gain.x_cl: self.test_X_batch, self.gain.t_img_for_IoU: self.test_img_Y_batch, self.gain.dropout_mode: False}
            )
            
            del self.idx_list_positive_only[0 : batch_size]

            self.IoU_list_on_batch.append(self.IoU)

        self.acc_res = sum(self.c_sum_list_on_batch) / len(self.idx_list_origin)
        self.loss_res = sum(self.loss_list_on_batch) / len(self.idx_list_origin)
        self.IoU_res = sum(self.IoU_list_on_batch) / len(self.idx_list_positive_only_origin) # positive data 分だけ扱う
        print('[Test results]: (Acc): {Acc}, (Loss): {Loss}, (IoU): {IoU}'.format(Acc = self.acc_res, Loss = self.loss_res, IoU = self.IoU_res))
