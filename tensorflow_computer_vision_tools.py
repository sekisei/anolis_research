# coding: UTF-8 
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
import random
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
import cv2
from tqdm import tqdm
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.losses import binary_crossentropy
import math

class computer_vision_tools():

    def __init__ (self):
        pass

    def get_Attention_Map(self, tensor_of_last_feature, VGG16_out):
        self.grad = tf.gradients(VGG16_out, tensor_of_last_feature)[0]
        self.alpha = tf.reduce_sum(self.grad, axis = [1, 2])
        self.Attention_Map = tf.multiply(tf.expand_dims(tf.expand_dims(self.alpha, 1), 1), tensor_of_last_feature)
        self.Attention_Map = tf.reduce_sum(self.Attention_Map, -1)
        self.zero = tf.zeros(tf.shape(self.Attention_Map))
        self.Attention_Map_Relu = tf.maximum(self.Attention_Map, self.zero)
        return self.Attention_Map_Relu

    def resize_Attention_Map(self, att_map):
        #self.anti_NaN = 1.0e-10
        self.size = [224, 224]
        #最大値で除算して画像処理向けにする
        #self.int_att_map = tf.divide(att_map, (tf.reduce_max(att_map) + self.anti_NaN)) * 255
        self.att_map_with_channel = tf.expand_dims(att_map, -1)
        self.att_map_resized = tf.image.resize_images(self.att_map_with_channel, self.size, method = tf.image.ResizeMethod.BILINEAR)
        #self.att_map_rescaled = tf.divide(self.att_map_resized, (tf.reduce_max(self.att_map_resized) + self.anti_NaN)) * 255
        return self.att_map_resized

    def resize_Attention_Map2(self, Attention_Map):
        self.result_on_batch = [np.uint8(Attention_Map[idx] / (Attention_Map[idx].max() + 1.0e-10) * 255.0) for idx in range(0, len(Attention_Map))]
        self.result_on_batch = np.array(self.result_on_batch)
        self.result_on_batch = [cv2.resize(np.expand_dims(self.result_on_batch[idx], axis = -1), (224, 224), interpolation = cv2.INTER_LANCZOS4) for idx in range(0, len(Attention_Map))]
        self.result_on_batch = np.array(self.result_on_batch)
        return self.result_on_batch

    def get_Heat_Map(self, Attention_Map):
        #self.result_on_batch = np.expand_dims(Attention_Map, axis = -1)
        #1.0e-10 -> Anti NaN
        self.result_on_batch = [np.uint8(Attention_Map[idx] / (Attention_Map[idx].max() + 1.0e-10) * 255.0) for idx in range(0, len(Attention_Map))]
        self.result_on_batch = np.array(self.result_on_batch)
        self.result_on_batch = [cv2.resize(np.expand_dims(self.result_on_batch[idx], axis = -1), (224, 224), interpolation = cv2.INTER_LANCZOS4) for idx in range(0, len(Attention_Map))]
        self.result_on_batch = np.array(self.result_on_batch)
        self.result_on_batch = [cv2.applyColorMap(self.result_on_batch[idx], cv2.COLORMAP_JET) for idx in range(0, len(Attention_Map))]
        self.result_on_batch = np.array(self.result_on_batch)
        self.result_on_batch = [cv2.cvtColor(self.result_on_batch[idx], cv2.COLOR_BGR2RGB) for idx in range(0, len(Attention_Map))]
        return np.array(self.result_on_batch)

    def get_Masked_img(self, Attention_Map, target):
        #print(Attention_Map[0])
        self.result_on_batch = Attention_Map
        self.result_on_batch = np.array( [self.result_on_batch[idx] / (self.result_on_batch[idx].max() + 1.0e-10) for idx in range(0, len(Attention_Map))] )
        self.result_on_batch = [cv2.resize(self.result_on_batch[idx], (224, 224), interpolation = cv2.INTER_LANCZOS4) for idx in range(0, len(Attention_Map))]
        self.result_on_batch = np.array(self.result_on_batch)
        self.T = ( lambda x: 1.0 / (1.0 + np.exp(-x)) )
        self.A = [cv2.merge((self.result_on_batch[idx], self.result_on_batch[idx], self.result_on_batch[idx])) for idx in range(0, len(Attention_Map))]
        self.A = np.array(self.A)
        self.I = np.array( [target[idx].reshape((224, 224, 3)) for idx in range(0, len(Attention_Map))] )
        self.w, self.sigma = 8.0, 0.5
        self.Ic = np.array( [self.I[idx] - np.multiply(self.T(self.w * (self.A[idx] - self.sigma)), self.I[idx]) for idx in range(0, len(Attention_Map))] )
        #self.Ic = np.array([Ic[idx] if (Ic[idx] == Ic[0]).all() == True else target[idx] for idx in range(0, len(Ic))])
        #[Ic[idx] if (Ic[idx] == Ic[0]).all() == False else print(A[idx]) for idx in range(0, len(Ic))]
        return self.Ic

    def get_Masked_img_tensor(self, Attention_map, target):
        self.w, self.sigma = tf.constant(0.3), tf.constant(0.5) #8.0, 0.5
        self.att_map_resized = self.resize_Attention_Map(Attention_map)
        self.adjusted_att_map = tf.multiply(self.w, tf.subtract(self.att_map_resized, self.sigma))
        self.sigmoid_output = tf.divide(tf.constant(1.0), tf.constant(1.0) + tf.exp(tf.negative(self.adjusted_att_map)))
        self.Ic = tf.subtract(target, tf.multiply(self.sigmoid_output, target))
        return self.Ic
    
    def save_AM(self, Heat_Map = None, target = None, dir_path = '', name_numbering = 0):
        self.Heat_Map = (np.float32(Heat_Map)  + target.reshape((224, 224, 3)) / 2.0)
        self.Heat_Map_img = array_to_img(self.Heat_Map)
        save_img(dir_path + 'Attention_Map_' + str(name_numbering)+ '.png', self.Heat_Map_img)
        return
    
