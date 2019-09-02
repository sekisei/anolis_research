import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img

path = '/media/kai/4tb/anolis_dataset_for_DL/'
all_data_X_size = 64840

#マスク付きテスト用画像作成 (トカゲのみ対象の処理)
def img_masking(imgs = None):
    (X, Y_img) = (imgs[0], imgs[1])
    #画素値の反転
    Y_img_reverse = np.abs(Y_img - Y_img.max())
    Y_img_reverse = Y_img_reverse / Y_img_reverse.max()
    X_masking = (Y_img_reverse * X).astype(np.int)
    return X_masking  

mem_X = np.memmap(path + 'data_X.dat', dtype = 'float16', mode = 'r', shape = (all_data_X_size, 224, 224, 3))
mem_X_mask = np.memmap(path + 'data_X_masking.dat', dtype = 'float16', mode = 'r', shape = (all_data_X_size, 224, 224, 3))
mem_img_Y = np.memmap(path + 'data_img_Y.dat', dtype = 'float16', mode = 'r', shape = (int(all_data_X_size / 2), 224, 224, 1))
mem_Y = np.memmap(path + 'data_Y.dat', dtype = 'float16', mode = 'r', shape = (all_data_X_size, 1))

#X_mask = img_masking(imgs = (mem_X[0], mem_img_Y[0]))
save_img('0.png', mem_X[100])
save_img('0L.png', mem_img_Y[100])
save_img('masking.png', mem_X_mask[100])
