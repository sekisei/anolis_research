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
    def __init__(self, sess = None, stream = 'S_cl', Dropout_rate = 0.3, learning_rate = 1.0e-8): #1.0e-8
        
        self.stream = stream
        self.my_common_tools = common_tools.tools()
        self.my_cv_tools = tensorflow_computer_vision_tools.computer_vision_tools()

        self.gain = gain.load(Dropout_rate = Dropout_rate)
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
        (self.train_X, self.train_Y, self.train_img_Y) = (input_data[0], input_data[1], input_data[2])
        (self.access_list_for_Scl, self.access_list_for_Se) = (access_list[0], access_list[1])

        self.acc_list_on_epoch = [0 for self.e in range(0, epochs)]
        self.IoU_list_on_epoch = [0 for self.e in range(0, epochs)]
        self.loss_list_on_epoch = [0 for self.e in range(0, epochs)]
        
        self.c_sum_list_on_batch = [0 for self.d_size in range(0, len(self.access_list_for_Scl))]
        self.IoU_sum_list_on_batch = [0 for self.d_size in range(0, len(self.access_list_for_Scl))]
        self.loss_sum_list_on_batch = [0 for self.d_size in range(0, len(self.access_list_for_Scl))]
        
        self.path_to_default_weights = '/media/kai/4tb/ckpt_data/default_weights/my_model.ckpt'
        self.path_to_weights = lambda epoch: '/media/kai/4tb/ckpt_data/'+ str(epoch) +'/my_model' + str(epoch) + '.ckpt'
        
        if save_default_weights == True: self.saver.save(self.sess, self.path_to_default_weights)

        if stream == 'S_cl':
            for self.epoch in range(0, epochs):
                print('epoch: ' + str(self.epoch))
                if save_weights == True: self.saver.save(self.sess, self.path_to_weights(self.epoch))
                random.shuffle(self.access_list_for_Scl)
                for self.idx, self.idx_Scl in enumerate(tqdm(self.access_list_for_Scl)):
                    self.train_X_batch_Scl = self.train_X[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size]
                    self.train_Y_batch = self.train_Y[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size].reshape((batch_size, 1))
                    [self.min_res, self.c_sum, self.l_sum] = self.sess.run(
                        [self.target, self.gain.correct_sum, self.gain.loss_sum_Lcl],
                        feed_dict = {self.gain.x_cl: self.train_X_batch_Scl, self.gain.t: self.train_Y_batch}
                    )
                    self.c_sum_list_on_batch[self.idx], self.loss_sum_list_on_batch[self.idx] = self.c_sum, self.l_sum
                    
                self.acc_list_on_epoch[self.epoch] = sum(self.c_sum_list_on_batch) / len(self.access_list_for_Scl)
                self.loss_list_on_epoch[self.epoch] = sum(self.loss_sum_list_on_batch) / len(self.access_list_for_Scl)
                print('Acc: {Acc}, Loss: {Loss}'.format(Acc = self.acc_list_on_epoch[self.epoch], Loss = self.loss_list_on_epoch[self.epoch]))

        if stream == 'S_cl and S_self':
            for self.epoch in range(0, epochs):
                print('epoch: ' + str(self.epoch))
                if save_weights == True: self.saver.save(self.sess, self.path_to_weights(self.epoch))
                random.shuffle(self.access_list_for_Scl)
                for self.idx, self.idx_Scl in enumerate(tqdm(self.access_list_for_Scl)):
                    self.train_X_batch_Scl = self.train_X[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size]
                    self.train_Y_batch = self.train_Y[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size].reshape((batch_size, 1))
                    self.Masked_val = self.gain.Masked_img.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.train_X_batch_Scl})
                    [self.min_res, self.c_sum, self.l_sum] = self.sess.run(
                        [self.target, self.gain.correct_sum, self.gain.loss_sum_Lself],
                        feed_dict = {self.gain.x_cl: self.train_X_batch_Scl, self.gain.x_masked: self.Masked_val, self.gain.t: self.train_Y_batch}
                    )
                    self.c_sum_list_on_batch[self.idx], self.loss_sum_list_on_batch[self.idx] = self.c_sum, self.l_sum
                    
                self.acc_list_on_epoch[self.epoch] = sum(self.c_sum_list_on_batch) / len(self.access_list_for_Scl)
                self.loss_list_on_epoch[self.epoch] = sum(self.loss_sum_list_on_batch) / len(self.access_list_for_Scl)
                print('Acc: {Acc}, Loss: {Loss}'.format(Acc = self.acc_list_on_epoch[self.epoch], Loss = self.loss_list_on_epoch[self.epoch]))                    

        if stream == 'S_cl, S_self and S_ext':
            for self.epoch in range(0, epochs):
                print('epoch: ' + str(self.epoch))

                #Attention Mapの保存
                self.img_num = 22
                self.INPUT = self.train_X[self.img_num].reshape((1, 224, 224, 3))
                self.Attention_Map_val = self.gain.resized_AM.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.INPUT}) ###
                self.HeatMap_val = self.my_cv_tools.get_Heat_Map(self.Attention_Map_val).reshape((224, 224, 3)) ###
                self.my_cv_tools.save_AM(Heat_Map = self.HeatMap_val, target = self.INPUT, name_numbering = self.epoch)
                save_img('Input'+str(self.epoch)+'.png', self.INPUT.reshape((224, 224, 3))) ###
                
                if save_weights == True: self.saver.save(self.sess, self.path_to_weights(self.epoch))
                random.shuffle(self.access_list_for_Scl)
                random.shuffle(self.access_list_for_Se)
                for self.idx, (self.idx_Scl, self.idx_Se) in enumerate( tqdm(zip(self.access_list_for_Scl, self.access_list_for_Se), total = len(self.access_list_for_Scl)) ):
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
                    [self.min_res, self.c_sum, self.IoU, self.l_sum] = self.sess.run(
                        [self.target, self.gain.correct_sum, self.gain.IoU, self.gain.loss_sum_Lext],
                        feed_dict = self.feed_dict
                    )
                    
                    self.IoU_sum_list_on_batch[self.idx] = self.IoU
                    self.c_sum_list_on_batch[self.idx], self.loss_sum_list_on_batch[self.idx] = self.c_sum, self.l_sum
                    
                self.acc_list_on_epoch[self.epoch] = sum(self.c_sum_list_on_batch) / len(self.access_list_for_Scl)
                self.loss_list_on_epoch[self.epoch] = sum(self.loss_sum_list_on_batch) / len(self.access_list_for_Scl)
                self.IoU_list_on_epoch[self.epoch] = sum(self.IoU_sum_list_on_batch) / len(self.access_list_for_Se)
                print(
                    'Acc: {Acc}, IoU: {IoU}, Loss: {Loss}'.format(
                        Acc = self.acc_list_on_epoch[self.epoch],
                        IoU = self.IoU_list_on_epoch[self.epoch],
                        Loss = self.loss_list_on_epoch[self.epoch])
                )
                
        print(self.acc_list_on_epoch)
        print(self.loss_list_on_epoch)
        print(self.IoU_list_on_epoch)

        return {'acc': self.acc_list_on_epoch, 'iou': self.IoU_list_on_epoch, 'loss': self.loss_list_on_epoch}

    def Test(self, batch_size = 1, access_list_for_test = None, input_data = None):
        self.data_X, self.data_Y = input_data[0], input_data[1]
        self.acc_res, self.loss_res = 0, 0
        self.Max = int(math.ceil(float(len(access_list_for_test)) / float(batch_size)))
        self.c_sum_list_on_batch = [0 for self.d_size in range(0, self.Max)]
        self.loss_list_on_batch = [0 for self.d_size in range(0, self.Max)]
        self.access_list_on_batch = []

        for self.idx in range(0, self.Max): self.access_list_on_batch.append(access_list_for_test[self.idx * batch_size : (self.idx + 1) * batch_size])
        
        for self.idx, self.batch in enumerate(tqdm(self.access_list_on_batch)):
            self.test_X_batch = np.array([self.data_X[self.data_idx] for self.data_idx in self.batch])
            self.test_Y_batch = np.array([self.data_Y[self.data_idx] for self.data_idx in self.batch]).reshape((len(self.test_X_batch), 1))
            [self.c_sum, self.l_sum] = self.sess.run(
                [self.gain.correct_sum, self.gain.loss_sum_Lcl],
                feed_dict = {self.gain.x_cl: self.test_X_batch, self.gain.t: self.test_Y_batch}
            )
            self.c_sum_list_on_batch[self.idx], self.loss_list_on_batch[self.idx] = self.c_sum, self.l_sum
            
        self.acc_res = sum(self.c_sum_list_on_batch) / len(access_list_for_test)
        self.loss_res = sum(self.loss_list_on_batch) / len(access_list_for_test)
        print('Acc: {Acc}, Loss: {Loss}'.format(Acc = self.acc_res, Loss = self.loss_res))



    
