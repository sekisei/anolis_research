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

import tensorflow as tf
import random
import numpy as np
from sklearn.model_selection import KFold

#custom library
import common_tools
import tensorflow_computer_vision_tools
import dataset_loader_for_K_hold_cross_validation
import experimental_program

#temp use
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img  

#環境設定
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

#3 streams: 'S_cl', 'S_cl and S_am', 'S_cl, S_am and S_e'
stream = 'S_cl, S_am and S_e'
#stream = 'S_cl'
epochs = 30
train_batch_size = 32
test_batch_size = 32
K = 4

tool = common_tools.tools()
exp_program = experimental_program.Set(stream = stream, learning_rate = 1.0e-6) #1.0e-8
loader = dataset_loader_for_K_hold_cross_validation.dataset_loader(path = '/media/kai/4tb/anolis_dataset_for_DL/')
dataset = loader.load_dataset() #X, Y, Y_img

KF = KFold(n_splits = K, shuffle = False)
all_index = KF.split(dataset['X'])

for k_num, (train_index, test_index) in enumerate(all_index):
    print('[Data size]: (Train)', len(train_index), ' (Test): ', len(test_index))
    print('[K num]    : ', k_num)

    random.shuffle(train_index)

    # streamごとにリストを分割。比率は自動的に判断する
    Scl_Sam_idx, Se_idx = loader.split_access_list_for_each_stream(stream = stream, train_idx = train_index, default_rate = (0.9, 0.1))

    # Stream externalにはpositiveデータインデックスだけ渡す
    Se_idx_positive_only = [idx for idx in Se_idx if dataset['Y'][idx] == 1]

    #Stream external用インデックスの拡張（index != 0であれば）
    Se_idx_expanded = loader.expand_index(index = Se_idx_positive_only, size = len(Scl_Sam_idx))

    exp_program.saver.restore(exp_program.sess, '/media/kai/4tb/ckpt_data/default_weights/my_model.ckpt')

    save_default_weights = False
    if k_num == 0: save_default_weights = True
    
    #--学習--
    #バッチ学習に対応させるのであれば、損失関数及びデータ分けに変更が必要である
    #access_list引数はストリームによって自動でrateが変わる
    hist = exp_program.Train(
        epochs = epochs,
        batch_size = train_batch_size,
        save_weights = True,
        save_default_weights = save_default_weights,
        stream = stream,
        access_list = (Scl_Sam_idx, Se_idx_expanded),
        input_data = (dataset['X'], dataset['Y'], dataset['Y_img'])
    )
    
    tool.save_history_as_txt(acc = np.array(hist['acc']), loss = np.array(hist['loss']), dir_path = '', numbering = k_num)

    # IoU評価にはpositiveデータインデックスだけ渡す
    idx_positive_only = [idx for idx in test_index if dataset['Y'][idx] == 1]

    #--試験--
    #試験時および評価時はバッチサイズ指定可
    #ドロップアウトは自動でオフになる
    exp_program.Test(
        batch_size = test_batch_size,
        idx_list_for_test = (test_index, idx_positive_only),
        input_data = dataset
    )    
