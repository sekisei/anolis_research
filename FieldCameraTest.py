# coding: UTF-8
from keras.preprocessing.image import load_img, img_to_array, array_to_img
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.backend import tensorflow_backend as backend
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Activation, Input
from keras.layers.core import Dropout, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.utils.vis_utils import plot_model
from multiprocessing import Process, Queue
from keras.applications.vgg16 import VGG16
from keras.models import Sequential,Model
from keras.models import model_from_json
from keras.optimizers import SGD,Adam
#from keras.preprocessing import image
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import keras
import time
import cv2
import gc
import os

class legacy_detection_process():
    
    def __init__(self, model, file_path, save_path):
        self.model = model
        self.save_path = save_path
        self.image_width = 224
        self.image_height = 224
        (self.crop_size_x, self.crop_size_y) = (224, 224)
        self.skip = 50
        self.img_anolis = Image.open(file_path)
        (self.height, self.width) = (self.img_anolis.height, self.img_anolis.width)
        self.img_for_rectangle = self.img_anolis.copy()
        self.img_for_crop = self.img_anolis.copy()
        self.draw = ImageDraw.Draw(self.img_for_rectangle)
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 64)

    def show_rectangle(self):    
        for self.x in range(0, self.width - self.crop_size_x):
            for self.y in range(0, self.height - self.crop_size_y):
                if self.skip * self.y + self.crop_size_y > self.height or self.skip * self.x + self.crop_size_x > self.width: break
                self.cropped_img = self.img_for_crop.crop((self.skip * self.x, self.skip * self.y, self.skip * self.x + self.crop_size_x, self.skip * self.y + self.crop_size_y))
                #cropped_img=cropped_img.resize((224,224))
                self.array_anolis = img_to_array(self.cropped_img)
                self.output = self.model.predict(np.reshape(self.array_anolis, (1, 224, 224, 3)), batch_size = 1)
                if self.output[0][0] > 0.9:
                    #draw.text((skip*x,skip*y),str(output[0][0]),font=font,fill=(255,0,0,128))
                    self.draw.rectangle((self.skip * self.x, self.skip * self.y, self.skip * self.x + self.crop_size_x, self.skip * self.y + self.crop_size_y), outline = (255, 0, 0))
                    self.draw.rectangle((self.skip * self.x + 1, self.skip * self.y + 1, self.skip * self.x + self.crop_size_x - 1, self.skip * self.y + self.crop_size_y - 1), outline = (255, 0, 0))
                    self.draw.rectangle((self.skip * self.x + 2, self.skip * self.y + 2, self.skip * self.x + self.crop_size_x - 2, self.skip * self.y + self.crop_size_y - 2), outline = (255, 0, 0))
                    self.draw.rectangle((self.skip * self.x + 3, self.skip * self.y + 3, self.skip * self.x + self.crop_size_x - 3, self.skip * self.y + self.crop_size_y - 3), outline = (255, 0, 0))
        self.img_for_rectangle.save(self.save_path)



        
'''
cap = cv2.VideoCapture('test.mp4')
ret, frame = cap.read()
(height,width)=frame.shape[:2]
(x_pos_list,y_pos_list)=(list(),list())
[x_pos_list.append(x*crop_size_x) for x in range(0,int(width/crop_size_x))]
[y_pos_list.append(y*crop_size_y) for y in range(0,int(height/crop_size_y))]
'''

'''
while True:
    ret, frame = cap.read()
    
    for x in x_pos_list:
        for y in y_pos_list:
            copied_frame=frame.copy()
            cropped_frame=copied_frame[y:y+crop_size_y,x:x+crop_size_x]
            cropped_frame=cv2.resize(cropped_frame,(224,224))
            RGBframe=cv2.cvtColor(cropped_frame,cv2.COLOR_BGR2RGB)
            img_array=img_to_array(RGBframe)
            img_array=img_array/255.0
            output=FT_model.predict(np.reshape(img_array,(1,224,224,3)),batch_size=1)
            #print(output[0][0])
            #if output[0][0]>0.9:
            #    frame = cv2.rectangle(frame,(x,y),(x+crop_size_x,y+crop_size_y),(0,0,255),1)
            cv2.imshow('window',show_rectangles(frame))
            #cv2.imshow('window',frame)
    
    copied_frame=frame.copy()
    cropped_frame=copied_frame[200:200+crop_size_y,760:760+crop_size_x]
    #cropped_frame=copied_frame[210:210+crop_size_y,0:crop_size_x]
    cropped_frame=cv2.resize(cropped_frame,(224,224))
    RGBframe=cv2.cvtColor(cropped_frame,cv2.COLOR_BGR2RGB)
    img_array=img_to_array(cropped_frame)
    img_array=img_array/255.0
    output=FT_model.predict(np.reshape(img_array,(1,224,224,3)),batch_size=1)
    print(output[0][0])
    cv2.imshow('window',cropped_frame)
    
    k = cv2.waitKey(1)
    if k == 27 :break

cap.release()
cv2.destroyAllWindows()
'''
backend.clear_session()
