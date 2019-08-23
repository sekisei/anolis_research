# coding: UTF-8
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
    def __init__(self, sess = None, stream = 'S_cl', Dropout_rate = 0.5, learning_rate = 1.0e-8):
        
        self.stream = stream
        self.my_common_tools = common_tools.tools()
        self.my_cv_tools = tensorflow_computer_vision_tools.computer_vision_tools()

        self.gain = gain.load(Dropout_rate = Dropout_rate)
        self.Loss_of_stream = self.gain.set_stream(stream = self.stream)
        self.target = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.Loss_of_stream)

        #予測結果・誤差集計用リスト
        self.acc_list_on_epoch = {'train': [], 'valid': [], 'test': []}
        self.loss_list_on_epoch = {'train': [], 'valid': [], 'test': []}
    
        self.sess = K.get_session()
        self.uninitialized_variables = [self.v for self.v in tf.global_variables() if not hasattr(self.v, '_keras_initialized') or not self.v._keras_initialized]
        #print(sess.run(tf.report_uninitialized_variables(tf.global_variables())))
        self.sess.run(tf.variables_initializer(self.uninitialized_variables))
        self.sess.run(tf.initializers.local_variables())

        self.saver = tf.train.Saver(max_to_keep = None)
    
    def result_processing(self, dropout_mode = None, mode = None, batch_size = 1, access_list = None, data_X = None, data_Y = None):
        print('Calclating loss and acc. [mode: ' + mode + ']')
        #count correct on batch for acc
        self.Is_equal = tf.equal(tf.to_float(tf.greater(self.gain.y_cl, 0.5)), tf.to_float(self.gain.t))
        self.correct_sum = tf.reduce_sum(tf.cast(self.Is_equal, tf.float32))
        self.loss_sum = tf.reduce_sum(self.gain.Lcl) #テスト時はLcl(Stream classificationの損失関数)のみ扱う

        self.sum_list = []
        self.loss_list = []

        self.access_list_on_batch = [self.idx for self.idx in range(0, int(math.ceil(float(len(access_list)) / float(batch_size))))]
        for self.idx in tqdm(self.access_list_on_batch):
            self.X_batch = data_X[self.idx * batch_size : (self.idx + 1) * batch_size]
            self.Y_batch = data_Y[self.idx * batch_size : (self.idx + 1) * batch_size]
            self.Y_batch = self.Y_batch.reshape((len(self.Y_batch), 1))
            self.sum_list.append(self.correct_sum.eval(session = self.sess, feed_dict = {self.gain.dropout_mode: dropout_mode, self.gain.x_cl: self.X_batch, self.gain.t: self.Y_batch}))
            self.loss_list.append(self.loss_sum.eval(session = self.sess, feed_dict = {self.gain.dropout_mode: dropout_mode, self.gain.x_cl: self.X_batch, self.gain.t: self.Y_batch}))
            
        self.acc_list_on_epoch[mode].append(sum(self.sum_list) / len(access_list))
        self.loss_list_on_epoch[mode].append(sum(self.loss_list) / len(access_list))
        print(self.acc_list_on_epoch)
        print(self.loss_list_on_epoch)
       
    #Scl, Sself, Sextで学習方法を分割
    def Train(self, epochs = 50, batch_size = 1, save_weights = True, save_default_weights = True, stream = None, access_list = None, input_data = None):
        (self.train_X, self.train_Y, self.train_img_Y) = (input_data[0], input_data[1], input_data[2])
        (self.access_list_for_Scl, self.access_list_for_Se) = (access_list[0], access_list[1])
        
        self.path_to_default_weights = '/media/kai/4tb/ckpt_data/default_weights/my_model.ckpt'
        self.path_to_weights = lambda epoch: '/media/kai/4tb/ckpt_data/'+ str(epoch) +'/my_model' + str(epoch) + '.ckpt'
        
        if save_default_weights == True: self.saver.save(self.sess, self.path_to_default_weights)

        if stream == 'S_cl':
            for self.epoch in range(0, epochs):
                print('epoch: ' + str(self.epoch))
                if save_weights == True: self.saver.save(self.sess, self.path_to_weights(self.epoch))
                random.shuffle(self.access_list_for_Scl)
                for self.idx_Scl in tqdm(self.access_list_for_Scl):
                    self.train_X_batch_Scl = self.train_X[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size]
                    self.train_Y_batch = self.train_Y[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size].reshape((batch_size, 1))
                    self.sess.run(self.target, feed_dict = {self.gain.x_cl: self.train_X_batch_Scl, self.gain.t: self.train_Y_batch})

        if stream == 'S_cl and S_self':
            for self.epoch in range(0, epochs):
                print('epoch: ' + str(self.epoch))
                if save_weights == True: self.saver.save(self.sess, self.path_to_weights(self.epoch))
                random.shuffle(self.access_list_for_Scl)
                for self.idx_Scl in tqdm(self.access_list_for_Scl):
                    self.train_X_batch_Scl = self.train_X[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size]
                    self.train_Y_batch = self.train_Y[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size].reshape((batch_size, 1))
                    self.Masked_val = self.gain.Masked_img.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.train_X_batch_Scl})
                    self.sess.run(self.target, feed_dict = {self.gain.x_cl: self.train_X_batch_Scl, self.gain.x_masked: self.Masked_val, self.gain.t: self.train_Y_batch})

        if stream == 'S_cl, S_self and S_ext':
            for self.epoch in range(0, epochs):
                print('epoch: ' + str(self.epoch))
                if save_weights == True: self.saver.save(self.sess, self.path_to_weights(self.epoch))
                random.shuffle(self.access_list_for_Scl)
                random.shuffle(self.access_list_for_Se)
                for (self.idx_Scl, self.idx_Se) in tqdm(zip(self.access_list_for_Scl, self.access_list_for_Se)):
                    self.train_X_batch_Scl = self.train_X[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size]
                    self.train_X_batch_Se = self.train_X[self.idx_Se * batch_size : (self.idx_Se + 1) * batch_size]
                    self.Masked_val = self.gain.Masked_img.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.train_X_batch_Scl})
                    self.train_Y_batch = self.train_Y[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size].reshape((batch_size, 1))
                    self.train_Y_img_batch = self.train_img_Y[self.idx_Se * batch_size : (self.idx_Se + 1) * batch_size] #/ 255.0
                    self.feed_dict = {
                        self.gain.x_cl: self.train_X_batch_Scl,
                        self.gain.x_masked: self.Masked_val,
                        self.gain.x_ext: self.train_X_batch_Se,
                        self.gain.t: self.train_Y_batch,
                        self.gain.t_img: self.train_Y_img_batch
                    }
                    self.sess.run(self.target, feed_dict = self.feed_dict)
        
        
    def Test(self, batch_size = 1, access_list_for_test = None, input_data = None):
        (self.test_X, self.test_Y) = (input_data[0], input_data[1])
        self.result_processing(dropout_mode = False, mode = 'test', batch_size = batch_size, access_list = access_list_for_test, data_X = self.test_X, data_Y = self.test_Y)

    #マスク付き画像データ用のテストメソッドがあると便利かも
    #あと、メソッド内のテンソルはgainモジュール内で書いたほうがいい。無駄にテンソルが増える可能性がある。
    def Test_on_masking_data(self, batch_size = 1, access_list_for_test = None, input_data = None):
        (self.test_X, self.test_Y) = (input_data[0], input_data[1])
        def result_processing_on_masking_data(dropout_mode = None, mode = None, batch_size = 1, access_list = None, data_X = None, data_Y = None):
            print('Calclating loss and acc. [mode: ' + mode + ']')
            #count correct on batch for acc
            Is_equal = tf.equal(tf.to_float(tf.greater(self.gain.y_cl, 0.5)), tf.to_float(self.gain.t))
            correct_sum = tf.reduce_sum(tf.cast(self.Is_equal, tf.float32))
            loss_sum = tf.reduce_sum(self.gain.Lcl) #テスト時はLcl(Stream classificationの損失関数)のみ扱う
            
            self.sum_list = []
            self.loss_list = []
            
            self.access_list_on_batch = [self.idx for self.idx in range(0, int(math.ceil(float(len(access_list)) / float(batch_size))))]
            for self.idx in tqdm(self.access_list_on_batch):
                self.X_batch = data_X[self.idx * batch_size : (self.idx + 1) * batch_size]
                self.Y_batch = data_Y[self.idx * batch_size : (self.idx + 1) * batch_size]
                self.Y_batch = self.Y_batch.reshape((len(self.Y_batch), 1))
                self.sum_list.append(self.correct_sum.eval(session = self.sess, feed_dict = {self.gain.dropout_mode: dropout_mode, self.gain.x_cl: self.X_batch, self.gain.t: self.Y_batch}))
                self.loss_list.append(self.loss_sum.eval(session = self.sess, feed_dict = {self.gain.dropout_mode: dropout_mode, self.gain.x_cl: self.X_batch, self.gain.t: self.Y_batch}))

                self.acc_list_on_epoch[mode].append(sum(self.sum_list) / len(access_list))
                self.loss_list_on_epoch[mode].append(sum(self.loss_list) / len(access_list))
                print(self.acc_list_on_epoch)
                print(self.loss_list_on_epoch)
                
        self.result_processing_on_masking_data(dropout_mode = False, mode = 'test', batch_size = batch_size, access_list = access_list_for_test, data_X = self.test_X, data_Y = self.test_Y)


