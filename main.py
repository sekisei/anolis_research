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

#coding: UTF-8

#----------------------------ToDo----------------------------
#学習曲線の出力に対応させる必要
#評価と試験に対応させる必要
#重みデータの保存が必要（ロードも）
#experimental_programで用いられるcommon_toolsのcollect_countを見なおしたほうがよい
#------------------------------------------------------------

from tqdm import tqdm
from tensorflow.python.keras import Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.losses import binary_crossentropy

#custom library
#from tfGAIN_tool import get_Attention_Map, get_Heat_Map, get_Masked_img, save_imgs, resize_Attention_Map, resize_Attention_Map2, get_Masked_img_tensor
#from general_tool import correct_count_on_batch, get_loss_on_batch, show_figure_of_history, save_history_as_txt, IoU
import common_tools
import tensorflow_computer_vision_tools
import dataset_loader
import gain
import experimental_program

#環境設定
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

stream = 'S_cl, S_self and S_ext'
epochs = 50
batch_size = 1

exp_program = experimental_program.Set(stream = stream, training_mode = True, Dropout_rate = 0.5, learning_rate = 1.0e-8, dataset_rate = 0.8)

sess = K.get_session()
saver = tf.train.Saver(max_to_keep = None)
uninitialized_variables = [v for v in tf.global_variables() if not hasattr(v, '_keras_initialized') or not v._keras_initialized]
#print(sess.run(tf.report_uninitialized_variables(tf.global_variables())))
sess.run(tf.variables_initializer(uninitialized_variables))
sess.run(tf.initializers.local_variables())

#バッチ学習に対応させるのであれば、損失関数及びデータ分けに変更が必要である
exp_program.Train(epochs = epochs, batch_size = batch_size)


'''
process_list = list(range(0, int(len(train_X) / batch_size)))
process_list_valid = list(range(0, int(len(valid_X) / batch_size)))
process_list_test = list(range(0, len(test_X)))
correct_list = np.zeros(shape = (int(len(train_X) / batch_size)))
correct_list_valid = np.zeros(shape = (int(len(valid_X) / batch_size)))
correct_list_test = np.zeros(shape = len(test_X))
each_loss_list = np.zeros(shape = (int(len(train_X) / batch_size), 1))
each_loss_list_valid = np.zeros(shape = (int(len(valid_X) / batch_size), 1))
each_loss_list_test = np.zeros(shape = (len(test_X), 1))
acc_list = [0 for i in range(0, epochs)]
loss_list = [0 for i in range(0, epochs)]
val_acc_list = [0 for i in range(0, epochs)]
val_loss_list = [0 for i in range(0, epochs)]

#gain->percent=0.8, no gain-> percent=1.0
process_list_for_Scl, process_list_for_Se = make_access_list_for_stream_ext(process_list, percent = 0.8)
#process_list_for_Seがサイズ0のときはとりあえずダミーを作る（gainを適用しない時）
if len(process_list_for_Se) == 0: process_list_for_Se = process_list_for_Scl[:]
#SeはSclより扱うデータ数が少ないのでサイズ調整（ランダムに構成）
process_list_for_Se_resized = process_list_for_Se * (int(len(process_list_for_Scl) / len(process_list_for_Se)) + 1)
process_list_for_Se = [process_list_for_Se_resized[idx_num] for idx_num, element in enumerate(process_list_for_Scl)]
print(len(process_list_for_Scl), len(process_list_for_Se))
'''

'''
#minibatch
#saver.restore(sess, '/media/kai/4tb/ckpt_data/'+ str(11) +'/my_model' + str(11) + '.ckpt')
#saver.save(sess, 'ckpt_data/default_weights/my_model.ckpt')
saver.restore(sess, 'ckpt_data/default_weights/my_model.ckpt')
for epoch in range(0, epochs):
    print('epoch: ' + str(epoch))
    random.shuffle(process_list_for_Scl)
    random.shuffle(process_list_for_Se)
    
    for (idx_Scl, idx_Se) in tqdm(zip(process_list_for_Scl, process_list_for_Se)):
        train_X_batch_Scl = train_X[idx_Scl * batch_size : (idx_Scl + 1) * batch_size]
        train_X_batch_Se = train_X[idx_Se * batch_size : (idx_Se + 1) * batch_size]
        train_Y_batch = train_Y[idx_Scl * batch_size : (idx_Scl + 1) * batch_size].reshape((batch_size, 1))
        train_Y_img_batch = train_img_Y[idx_Se * batch_size : (idx_Se + 1) * batch_size] #/ 255.0
        #train_Y_img_batch_reverse = np.abs(train_Y_img_batch - train_Y_img_batch.max()) #画素値の反転
        #print(train_Y_batch)
        #save_img(str(idx_Se)+'.png', train_X_batch_Se.reshape(224,224,3))
        #save_img(str(idx_Se)+'_label.png', train_Y_img_batch.reshape(224,224,1))
        #class_num = 0
        #if np.all(train_Y_img_batch == 1.0): class_num = 1
        #Attention_Map_val = Attention_Map.eval(session = sess, feed_dict = {x_cl: train_X_batch_Scl})
        #HeatMap_val = get_Heat_Map(Attention_Map_val)
        Masked_val = Masked_img.eval(session = sess, feed_dict = {x_cl: train_X_batch_Scl})
        #save_imgs(HeatMap_val, Masked_val, train_X_batch_Se, name_numbering = idx_Se)
        #prob = y_cl.eval(session = sess, feed_dict = {x_cl: train_X_batch_Scl})
        #print('y_cl')
        #print(prob)
        sess.run(Objective_Lext, feed_dict = {x_cl: train_X_batch_Scl, x_masked: Masked_val, x_ext: train_X_batch_Se, t: train_Y_batch, t_img: train_Y_img_batch})
        #correct_list[idx] = correct_counter.eval(session = sess, feed_dict = {x_cl: train_X_batch, t: train_Y_batch})
        #each_loss_list[idx] = Lself.eval(session = sess, feed_dict = {x_cl: train_X_batch, x_masked: Masked_val, t: train_Y_batch})
    
    #for idx in tqdm(process_list_valid):
    #    valid_X_batch = valid_X[idx * batch_size : (idx + 1) * batch_size]
    #    valid_Y_batch = valid_Y[idx * batch_size : (idx + 1) * batch_size].reshape((batch_size, 2))
    #    correct_list_valid[idx] = correct_counter.eval(session = sess, feed_dict = {x_cl: valid_X_batch, t: valid_Y_batch})
    #    each_loss_list_valid[idx] = Lself.eval(session = sess, feed_dict = {x_cl: valid_X_batch, x_masked: Masked_val, t: valid_Y_batch})

    #acc_list[epoch] = correct_list.sum() / len(train_X)
    #val_acc_list[epoch] = correct_list_valid.sum() / len(valid_X)
    #loss_list[epoch] = each_loss_list.sum() / len(train_X)
    #val_loss_list[epoch] = each_loss_list_valid.sum() / len(valid_X)
    
    saver.save(sess, 'ckpt_data/'+ str(epoch) +'/my_model' + str(epoch) + '.ckpt')
    
    for idx in range(int(len(valid_X)/2-16), int(len(valid_X)/2+16)):
        range_start, range_end = idx, idx + 1
        valid_X_batch = valid_X[range_start : range_end]
        valid_Y_batch = valid_Y[range_start : range_end] 
        Attention_Map_val = Attention_Map.eval(session = sess, feed_dict = {x_cl: valid_X_batch})
        HeatMap_val = get_Heat_Map(Attention_Map_val)
        Masked_val = Masked_img.eval(session = sess, feed_dict = {x_cl: valid_X_batch})
        save_imgs(HeatMap_val, Masked_val, valid_X_batch, dir_path = 'pngs/',name_numbering = idx)
        prob = y_cl.eval(session = sess, feed_dict = {x_cl: valid_X_batch})
        print('y_cl')
        print(prob)
    
#save_history_as_txt(np.array(loss_list), np.array(acc_list), np.array(val_loss_list), np.array(val_acc_list), dir_path = 'hist/')
#show_figure_of_history(np.array(loss_list), np.array(acc_list), np.array(val_loss_list), np.array(val_acc_list), dir_path = 'hist/')

#test
saver.restore(sess, 'ckpt_data/'+ str(49) +'/my_model' + str(49) + '.ckpt')
for idx in tqdm(process_list_test):
    class_num = 1
    test_X_batch = test_X[idx : (idx + 1)]
    test_Y_batch = test_Y[idx : (idx + 1)].reshape((1, 1))
    #マスク付きテスト用画像作成 (トカゲのみ対象の処理)
    if idx < int(len(process_list_test) / 2): 
        test_Y_img_batch = test_img_Y[idx : (idx + 1)]
        test_Y_img_batch_reverse = np.abs(test_Y_img_batch - test_Y_img_batch.max()) #画素値の反転
        test_Y_img_batch_reverse = test_Y_img_batch_reverse / test_Y_img_batch_reverse.max()
        #test_X_batch = (test_Y_img_batch_reverse * test_X_batch).astype(np.int)
    
    correct_list_test[idx] = correct_counter.eval(session = sess, feed_dict = {x_cl: test_X_batch, t: test_Y_batch})
    each_loss_list_test[idx] = Lcl.eval(session = sess, feed_dict = {x_cl: test_X_batch, t: test_Y_batch})
    #print('ycl')
    #print(y_cl.eval(session = sess, feed_dict = {x_cl: test_X_batch, t: test_Y_batch}))
    #print(Lcl.eval(session = sess, feed_dict = {x_cl: test_X_batch, t: test_Y_batch}))
    Attention_Map_val = Attention_Map.eval(session = sess, feed_dict = {x_cl: test_X_batch})
    HeatMap_val = get_Heat_Map(Attention_Map_val)
    Masked_val = Masked_img.eval(session = sess, feed_dict = {x_cl: test_X_batch})
    save_imgs(HeatMap_val, Masked_val, test_X_batch, dir_path = 'pngs/', name_numbering = idx)

print(correct_list_test.sum() / len(test_X) / class_num)
print(each_loss_list_test.sum() / len(test_X) / class_num)
'''



