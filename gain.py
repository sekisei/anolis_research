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
#from tensorflow.python.keras.layers import Flatten, Dense, Activation, Dropout
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.losses import binary_crossentropy

#custom library
import tensorflow_computer_vision_tools

class load():
    def __init__ (self, Dropout_rate = 0.5):

        self.my_cv_tools = tensorflow_computer_vision_tools.computer_vision_tools()

        self.VGG16_No_Top = tf.keras.applications.vgg16.VGG16(weights = 'imagenet', include_top = False)
        for self.layer in self.VGG16_No_Top.layers[:11]:
            self.layer.trainable = False #Do not change
            print(str(self.layer) + ' <- Freeze')

        self.dropout_mode = tf.placeholder_with_default(True, shape = ())
            
        self.x_cl = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
        self.x_masked = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
        self.x_ext = tf.placeholder(tf.float32, shape = [None, 224, 224, 3])
        self.output_cl = self.VGG16_No_Top(self.x_cl)
        self.output_masked = self.VGG16_No_Top(self.x_masked)
        self.output_ext = self.VGG16_No_Top(self.x_ext)

        with tf.variable_scope('fully_connected'):
            self.GAP_cl = tf.keras.layers.GlobalAveragePooling2D(name = 'avg_pool')(self.output_cl)
            self.h1_cl = tf.layers.dense(self.GAP_cl, 512, activation = tf.nn.relu, name = 'dense1')
            self.D1_cl = tf.layers.dropout(self.h1_cl, rate = Dropout_rate, training = self.dropout_mode, name = 'dropout1')
            self.h2_cl = tf.layers.dense(self.D1_cl, 256, activation = tf.nn.relu, name = 'dense2')
            self.D2_cl = tf.layers.dropout(self.h2_cl, rate = Dropout_rate, training = self.dropout_mode, name = 'dropout2')
            self.VGG16_out_cl = tf.layers.dense(self.D2_cl, 1, activation = tf.nn.sigmoid, name = 'out')

        with tf.variable_scope('fully_connected', reuse = True):
            self.GAP_masked = tf.keras.layers.GlobalAveragePooling2D(name = 'avg_pool')(self.output_masked)
            self.h1_masked = tf.layers.dense(self.GAP_masked, 512, activation = tf.nn.relu, name = 'dense1')
            self.D1_masked = tf.layers.dropout(self.h1_masked, rate = Dropout_rate, training = self.dropout_mode, name = 'dropout1')
            self.h2_masked = tf.layers.dense(self.D1_masked, 256, activation = tf.nn.relu, name = 'dense2')
            self.D2_masked = tf.layers.dropout(self.h2_masked, rate = Dropout_rate, training = self.dropout_mode, name = 'dropout2')
            self.VGG16_out_masked = tf.layers.dense(self.D2_masked, 1, activation = tf.nn.sigmoid, name = 'out')

        with tf.variable_scope('fully_connected', reuse = True):
            self.GAP_ext = tf.keras.layers.GlobalAveragePooling2D(name = 'avg_pool')(self.output_ext)
            self.h1_ext = tf.layers.dense(self.GAP_ext, 512, activation = tf.nn.relu, name = 'dense1')
            self.D1_ext = tf.layers.dropout(self.h1_ext, rate = Dropout_rate, training = self.dropout_mode, name = 'dropout1')
            self.h2_ext = tf.layers.dense(self.D1_ext, 256, activation = tf.nn.relu, name = 'dense2')
            self.D2_ext = tf.layers.dropout(self.h2_ext, rate = Dropout_rate, training = self.dropout_mode, name = 'dropout2')
            self.VGG16_out_ext = tf.layers.dense(self.D2_ext, 1, activation = tf.nn.sigmoid, name = 'out')

        #以下はスコープの番号に注意
        self.t = tf.placeholder(tf.float32, shape = (None, 1))
        self.t_img = tf.placeholder(tf.float32, shape = (None, 224, 224, 1))
        #t_img_reverse = tf.placeholder(tf.float32, shape = (None, 224, 224, 1))
        self.y_cl = self.VGG16_out_cl
        self.y_ext = self.VGG16_out_ext
        self.y_masked = self.VGG16_out_masked

        #[print(n.name) for n in tf.get_default_graph().as_graph_def().node]

        self.Graph = tf.get_default_graph()
        self.y_cl_logits = self.Graph.get_tensor_by_name('fully_connected/out/BiasAdd:0')
        self.y_masked_logits = self.Graph.get_tensor_by_name('fully_connected_1/out/BiasAdd:0') #Not to be used
        self.y_ext_logits = self.Graph.get_tensor_by_name('fully_connected_2/out/BiasAdd:0')

        #[0][0] -> anolis, [0][1] -> others
        self.block5_conv3_tensor = self.Graph.get_tensor_by_name('vgg16/block5_conv3/Relu:0')
        self.block5_conv3_tensor_ext = self.Graph.get_tensor_by_name('vgg16/block5_conv3_2/Relu:0') 
        self.Attention_Map = self.my_cv_tools.get_Attention_Map(self.block5_conv3_tensor, self.y_cl_logits)
        self.Attention_Map_ext = self.my_cv_tools.get_Attention_Map(self.block5_conv3_tensor_ext, self.y_ext_logits)
        self.resized_AM = self.my_cv_tools.resize_Attention_Map(self.Attention_Map)
        self.resized_AM_ext = self.my_cv_tools.resize_Attention_Map(self.Attention_Map_ext)
        self.Masked_img = self.my_cv_tools.get_Masked_img_tensor(self.Attention_Map, self.x_cl)
        (self.y_pred, self.y_label) = (self.y_cl, self.t)
        #correct_counter = correct_count_on_batch(y_pred, y_label) #very slow

        self.Lcl = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.t, logits = self.y_cl_logits)
        self.Lam = self.y_masked
        self.Le = tf.square(self.resized_AM_ext - self.t_img)
        self.alpha = tf.constant(1.0)
        self.omega = tf.constant(10.0)
        self.Lself = self.Lcl + tf.multiply(self.Lam, self.alpha)
        self.Lext = self.Lcl + tf.multiply(self.Lam, self.alpha) + tf.multiply(self.Le, self.omega)
        #Objective_Lext = tf.train.AdamOptimizer(learning_rate = 1.0e-8).minimize(Lext) #gain-> Lext, no gain-> Lcl

    def set_stream(self, stream = 'S_cl'):
        if stream == 'S_cl': return self.Lcl
        if stream == 'S_cl and S_self': return self.Lself
        if stream == 'S_cl, S_self and S_ext': return self.Lext
