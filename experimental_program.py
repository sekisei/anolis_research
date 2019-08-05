# coding: UTF-8 
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
import random
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
import cv2
#import os

from tqdm import tqdm
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.losses import binary_crossentropy

#custom library
import common_tools
import tensorflow_computer_vision_tools
import dataset_loader
import gain
    
class Set():
    def __init__(self, stream = 'S_cl', training_mode = True, Dropout_rate = 0.5, learning_rate = 1.0e-8, dataset_rate = 0.8):
        #saver.restore(sess, 'ckpt_data/default_weights/my_model.ckpt')
        self.stream = stream

        self.my_common_tools = common_tools.tools()
        self.my_cv_tools = tensorflow_computer_vision_tools.computer_vision_tools()
        self.dataset = dataset_loader.load(base_path = '/home/kai/anolis/dataset/npy_dataset_0/')

        self.gain = gain.load(training_mode = training_mode, Dropout_rate = Dropout_rate)
        self.Loss_of_stream = self.gain.set_stream(stream = self.stream)
        self.target = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.Loss_of_stream)

        if self.stream == 'S_cl, S_self and S_ext':
            self.access_list_for_Scl, self.access_list_for_Se = self.dataset.make_access_list(self.dataset.train_X, dataset_rate = dataset_rate, stream = self.stream)
        else:
            self.access_list_for_Scl = self.dataset.make_access_list(self.dataset.train_X, stream = self.stream)
            
        self.sess = K.get_session()
        self.saver = tf.train.Saver(max_to_keep = None)
        self.uninitialized_variables = [self.v for self.v in tf.global_variables() if not hasattr(self.v, '_keras_initialized') or not self.v._keras_initialized]
        #print(sess.run(tf.report_uninitialized_variables(tf.global_variables())))
        self.sess.run(tf.variables_initializer(self.uninitialized_variables))
        self.sess.run(tf.initializers.local_variables())

    #Scl, Sself, Sextで学習方法を分割
    def Train(self, epochs = 50, batch_size = 1):
        if self.stream == 'S_cl':
            for self.epoch in range(0, epochs):
                random.shuffle(self.access_list_for_Scl)
                for self.idx_Scl in tqdm(self.access_list_for_Scl):
                    self.train_X_batch_Scl = self.dataset.train_X[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size]
                    self.train_Y_batch = self.dataset.train_Y[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size].reshape((batch_size, 1))
                    self.sess.run(self.target, feed_dict = {self.gain.x_cl: self.train_X_batch_Scl, self.gain.t: self.train_Y_batch})

        if self.stream == 'S_cl and S_self':
            for self.epoch in range(0, epochs):
                random.shuffle(self.access_list_for_Scl)
                for self.idx_Scl in tqdm(self.access_list_for_Scl):
                    self.train_X_batch_Scl = self.dataset.train_X[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size]
                    self.train_Y_batch = self.dataset.train_Y[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size].reshape((batch_size, 1))
                    self.Masked_val = self.gain.Masked_img.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.train_X_batch_Scl})
                    self.sess.run(self.target, feed_dict = {self.gain.x_cl: self.train_X_batch_Scl, self.gain.x_masked: self.Masked_val, self.gain.t: self.train_Y_batch})

        if self.stream == 'S_cl, S_self and S_ext':
            for self.epoch in range(0, epochs):
                random.shuffle(self.access_list_for_Scl)
                random.shuffle(self.access_list_for_Se)
                for (self.idx_Scl, self.idx_Se) in tqdm(zip(self.access_list_for_Scl, self.access_list_for_Se)):
                    self.train_X_batch_Scl = self.dataset.train_X[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size]
                    self.train_X_batch_Se = self.dataset.train_X[self.idx_Se * batch_size : (self.idx_Se + 1) * batch_size]
                    self.Masked_val = self.gain.Masked_img.eval(session = self.sess, feed_dict = {self.gain.x_cl: self.train_X_batch_Scl})
                    self.train_Y_batch = self.dataset.train_Y[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size].reshape((batch_size, 1))
                    self.train_Y_img_batch = self.dataset.train_img_Y[self.idx_Se * batch_size : (self.idx_Se + 1) * batch_size] #/ 255.0
                    self.feed_dict = {
                        self.gain.x_cl: self.train_X_batch_Scl,
                        self.gain.x_masked: self.Masked_val,
                        self.gain.x_ext: self.train_X_batch_Se,
                        self.gain.t: self.train_Y_batch,
                        self.gain.t_img: self.train_Y_img_batch
                    }
                    self.sess.run(self.target, feed_dict = self.feed_dict)

    def Test(self, batch_size = 1):
        for self.epoch in range(0, epochs):
            for self.idx_Scl in tqdm(self.access_list_for_Scl):
                self.test_X_batch_Scl = self.dataset.test_X[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size]
                self.test_Y_batch = self.dataset.test_Y[self.idx_Scl * batch_size : (self.idx_Scl + 1) * batch_size].reshape((batch_size, 1))
                #self.correct_list_test[idx] = correct_counter.eval(session = sess, feed_dict = {x_cl: test_X_batch, t: test_Y_batch})
                #self.each_loss_list_test[idx] = Lcl.eval(session = sess, feed_dict = {x_cl: test_X_batch, t: test_Y_batch})
                #print('ycl')
                #print(y_cl.eval(session = sess, feed_dict = {x_cl: test_X_batch, t: test_Y_batch}))
                #print(Lcl.eval(session = sess, feed_dict = {x_cl: test_X_batch, t: test_Y_batch}))  
                #self.sess.run(self.target, feed_dict = {self.gain.x_cl: self.test_X_batch_Scl, self.gain.t: self.test_Y_batch})
        
                        


