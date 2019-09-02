# coding: UTF-8

#----------------------------memo----------------------------
#学習曲線の出力に対応させたいが、SselfやSextの組み合わせのせいで正解率の評価が難しい
#loss_sumを柔軟にしてfeed_dictもいじる？
#Sextは交差検証に向いていない（画像ラベルを全て作る必要があるためである）
#交差検証をするのであれば、SselfとSclの組み合わせしか選べない
#結論：Sextではホールドアウト検証を行い、SclとSselfの組み合わせでは交差検証を行う
#画像ラベルは植物も追加してみることにした。それでうまくいかなければ別の方法を考える。
#テスト方法を改造する必要がある
#------------------------------------------------------------

#-----------------------------data-----------------------------
#split_access_list_for_stream_extは以下のデータ構造でのみ使用可

#  Data_X.dat
#|aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa|ooooooooooooooooooooooooooooo|

#  Data_Y.dat
#|aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa|ooooooooooooooooooooooooooooo|

#  Data_img_Y.dat
#|aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa|          - empty -          |

# a ---> anolis, o ---> others
#--------------------------------------------------------------

import tensorflow as tf
import random
import numpy as np

#custom library
import common_tools
import tensorflow_computer_vision_tools
import dataset_loader_for_K_hold_cross_validation
import experimental_program

#temp use
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img  

#環境設定
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#3 streams: 'S_cl', 'S_cl and S_self', 'S_cl, S_self and S_ext'
stream = 'S_cl, S_self and S_ext'
#stream = 'S_cl' 
data_size = 64840
epochs = 50
train_batch_size = 1
test_batch_size = 32
data_rate = 0.95
K = 4

tool = common_tools.tools()
exp_program = experimental_program.Set(stream = stream, Dropout_rate = 0.3, learning_rate = 1.0e-8)
loader = dataset_loader_for_K_hold_cross_validation.dataset_loader(base_path = '/media/kai/4tb/anolis_dataset_for_DL/', data_size = data_size)

access_list_for_Scl_Sam, access_list_for_Se = loader.split_access_list_for_each_stream(
        stream = stream,
        access_list = [idx for idx in range(0, data_size)],
        data_Y = loader.data_Y,
        rate = loader.change_data_rate(stream = stream, rate = data_rate)
    )

#print('Scl_Scm')
#print(access_list_for_Scl_Sam)
#for i in range(32419, 32439): print(loader.data_Y[i])
#print('Se')
#print(access_list_for_Se)

splitted_dataset = loader.split_list_into_K_access_list(K = K, access_list = access_list_for_Scl_Sam, shuffle = True)

#print('plitted')
#for i in range(0, len(splitted_dataset)):
#    print('')
#    print(splitted_dataset[i])

for k_num in range(0, K):
    K_hold_access_list = loader.get_new_access_list(k_num = k_num, splitted_dataset = splitted_dataset, shuffle = False)
    (train_data_access_list, test_data_access_list) = (K_hold_access_list[0], K_hold_access_list[1])
    print('[Data size] Train: ' + str(len(train_data_access_list)) + ', Test: ' + str(len(test_data_access_list)))

    exp_program.saver.restore(exp_program.sess, '/media/kai/4tb/ckpt_data/default_weights/my_model.ckpt')
    
    '''
    #--学習--
    #バッチ学習に対応させるのであれば、損失関数及びデータ分けに変更が必要である
    #access_list引数はストリームによって自動でrateが変わる
    save_default_weights = False
    if k_num == 0: save_default_weights = True
    hist = exp_program.Train(
        epochs = epochs,
        batch_size = train_batch_size,
        save_weights = True,
        save_default_weights = save_default_weights,
        stream = stream,
        access_list = (train_data_access_list, access_list_for_Se),
        input_data = (loader.data_X, loader.data_Y, loader.data_img_Y)
    )
    tool.save_history_as_txt(acc = np.array(hist['acc']), loss = np.array(hist['loss']), iou_acc = np.array(hist['iou']), dir_path = '')
    '''
    
    #--試験--
    #試験時および評価時はバッチサイズ指定可
    #ドロップアウトは自動でオフになる
    exp_program.Test(
        batch_size = test_batch_size,
        access_list_for_test = test_data_access_list,
        input_data = (loader.data_X, loader.data_Y)
    )

    #マスク付き画像でのテスト
    exp_program.Test(
        batch_size = test_batch_size,
        access_list_for_test = test_data_access_list,
        input_data = (loader.data_X_mask, loader.data_Y)
    )
    #'''

'''
exp_program = experimental_program.Set(stream = stream, Dropout_rate = 0.5, learning_rate = 1.0e-8)
loader = dataset_loader_for_K_hold_cross_validation.dataset_loader(base_path = '/media/kai/4tb/anolis_dataset_for_DL/', data_size = data_size)
splitted_dataset = loader.split_list_into_K_access_list(K = 5, data_size = data_size, shuffle = True)

for k_num in range(0, 5):
    K_hold_access_list = loader.get_new_access_list(k_num = k_num, splitted_dataset = splitted_dataset, shuffle = False)
    (train_data_access_list, test_data_access_list) = (K_hold_access_list[0], K_hold_access_list[1])
    print('[Data size] Train: ' + str(len(train_data_access_list)) + ', Test: ' + str(len(test_data_access_list)))
    access_list_for_stream_ext = loader.split_access_list_for_stream_ext(
        access_list = train_data_access_list,
        data_Y = loader.data_Y,
        rate = loader.change_data_rate(stream = stream, rate = data_rate)
    )

    #--学習--
    #バッチ学習に対応させるのであれば、損失関数及びデータ分けに変更が必要である
    #access_list引数はストリームによって自動でrateが変わる
    save_default_weights = False
    if k_num == 0: save_default_weights = True
    exp_program.Train(
        epochs = epochs,
        batch_size = train_batch_size,
        save_weights = True,
        save_default_weights = save_default_weights,
        stream = stream,
        access_list = access_list_for_stream_ext,
        input_data = (loader.data_X, loader.data_Y, loader.data_img_Y)
    )

    #--試験--
    #試験時および評価時はバッチサイズ指定可
    #ドロップアウトは自動でオフになる
    #マスクつき画像
    exp_program.Test(
        batch_size = test_batch_size,
        access_list_for_test = test_data_access_list,
        input_data = (loader.data_X, loader.data_Y)
    )
'''

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



