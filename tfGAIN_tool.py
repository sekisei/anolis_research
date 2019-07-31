# coding: UTF-8 
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.keras import backend as K
from keras.applications.vgg16 import preprocess_input, decode_predictions
import random
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import cv2
from tqdm import tqdm
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.losses import binary_crossentropy
import math

def get_Attention_Map(tensor_of_last_feature, VGG16_out):
    grad = tf.gradients(VGG16_out, tensor_of_last_feature)[0]
    alpha = tf.reduce_sum(grad, axis = [1, 2])
    Attention_Map = tf.multiply(tf.expand_dims(tf.expand_dims(alpha, 1), 1), tensor_of_last_feature)
    Attention_Map = tf.reduce_sum(Attention_Map, -1)
    zero = tf.zeros(tf.shape(Attention_Map))
    Attention_Map_Relu = tf.maximum(Attention_Map, zero)
    return Attention_Map_Relu

def resize_Attention_Map(att_map):
    #anti_NaN = 1.0e-10
    size = [224, 224]
    #最大値で除算して画像処理向けにする
    #int_att_map = tf.divide(att_map, (tf.reduce_max(att_map) + anti_NaN)) * 255
    att_map_with_channel = tf.expand_dims(att_map, -1)
    att_map_resized = tf.image.resize_images(att_map_with_channel, size, method = tf.image.ResizeMethod.BILINEAR)
    #att_map_rescaled = tf.divide(att_map_resized, (tf.reduce_max(att_map_resized) + anti_NaN)) * 255
    return att_map_resized

def resize_Attention_Map2(Attention_Map):
    result_on_batch = [np.uint8(Attention_Map[idx] / (Attention_Map[idx].max() + 1.0e-10) * 255.0) for idx in range(0, len(Attention_Map))]
    result_on_batch = np.array(result_on_batch)
    result_on_batch = [cv2.resize(np.expand_dims(result_on_batch[idx], axis = -1), (224, 224), interpolation = cv2.INTER_LANCZOS4) for idx in range(0, len(Attention_Map))]
    result_on_batch = np.array(result_on_batch)
    return result_on_batch

def get_Heat_Map(Attention_Map):
    #result_on_batch = np.expand_dims(Attention_Map, axis = -1)
    #1.0e-10 -> Anti NaN
    result_on_batch = [np.uint8(Attention_Map[idx] / (Attention_Map[idx].max() + 1.0e-10) * 255.0) for idx in range(0, len(Attention_Map))]
    result_on_batch = np.array(result_on_batch)
    result_on_batch = [cv2.resize(np.expand_dims(result_on_batch[idx], axis = -1), (224, 224), interpolation = cv2.INTER_LANCZOS4) for idx in range(0, len(Attention_Map))]
    result_on_batch = np.array(result_on_batch)
    result_on_batch = [cv2.applyColorMap(result_on_batch[idx], cv2.COLORMAP_JET) for idx in range(0, len(Attention_Map))]
    result_on_batch = np.array(result_on_batch)
    result_on_batch = [cv2.cvtColor(result_on_batch[idx], cv2.COLOR_BGR2RGB) for idx in range(0, len(Attention_Map))]
    return np.array(result_on_batch)

def get_Masked_img(Attention_Map, target):
    #print(Attention_Map[0])
    result_on_batch = Attention_Map
    result_on_batch = np.array( [result_on_batch[idx] / (result_on_batch[idx].max() + 1.0e-10) for idx in range(0, len(Attention_Map))] )
    result_on_batch = [cv2.resize(result_on_batch[idx], (224, 224), interpolation = cv2.INTER_LANCZOS4) for idx in range(0, len(Attention_Map))]
    result_on_batch = np.array(result_on_batch)
    T = ( lambda x: 1.0 / (1.0 + np.exp(-x)) )
    A = [cv2.merge((result_on_batch[idx], result_on_batch[idx], result_on_batch[idx])) for idx in range(0, len(Attention_Map))]
    A = np.array(A)
    I = np.array( [target[idx].reshape((224, 224, 3)) for idx in range(0, len(Attention_Map))] )
    w, sigma = 8.0, 0.5
    Ic = np.array( [I[idx] - np.multiply(T(w * (A[idx] - sigma)), I[idx]) for idx in range(0, len(Attention_Map))] )
    #Ic = np.array([Ic[idx] if (Ic[idx] == Ic[0]).all() == True else target[idx] for idx in range(0, len(Ic))])
    #[Ic[idx] if (Ic[idx] == Ic[0]).all() == False else print(A[idx]) for idx in range(0, len(Ic))]
    return Ic

def get_Masked_img_tensor(Attention_map, target):
    w, sigma = tf.constant(0.3), tf.constant(0.5) #8.0, 0.5
    att_map_resized = resize_Attention_Map(Attention_map)
    adjusted_att_map = tf.multiply(w, tf.subtract(att_map_resized, sigma))
    sigmoid_output = tf.divide(tf.constant(1.0), tf.constant(1.0) + tf.exp(tf.negative(adjusted_att_map)))
    Ic = tf.subtract(target, tf.multiply(sigmoid_output, target))
    return Ic

def save_imgs(Heat_Map, Ic, target, idx = 0, dir_path = '', name_numbering = 0):
    Heat_Map = (np.float32(Heat_Map[idx])  + target[idx].reshape((224, 224, 3)) / 2.0)
    Heat_Map_img = array_to_img(Heat_Map)
    Ic = np.float32(Ic[idx])
    Ic_img = array_to_img(Ic)
    #Heat_Map_img.show()
    Heat_Map_img.save(dir_path + 'Attention_Map_' + str(name_numbering)+ '.png')
    Ic_img.save(dir_path + 'MASK_' + str(name_numbering)+ '.png')
