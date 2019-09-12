#coding: UTF-8

#Scl, SamのデータはSeのデータサイズがPositive分だけなのでバランスを保つためにNegative分追加削除される
#最終的にはPositive, Negativeバランスよく分割できるように交互に配置する
#(Example)
# (1)
# |-------------------------------------1.0------------------------------------------|
# |----------0.4----------|------0.1-----|----------0.4----------|--------0.1--------|
# |---Positive(Scl,Sam)---|-Positive(Se)-|---Negative(Scl,Sam)---|-Negative(deleted)-|
#
#                                          |
#                                          V
# (2)
# |----------------------0.8----------------------|
# |----------0.4----------|----------0.4----------|
# |---Positive(Scl,Sam)---|---Negative(Scl,Sam)---|
#
# |------0.1-----|
# |-Positive(Se)-|
#
#                         |
#                         V
# (3)
# |----------------------0.8----------------------|
# |PNPNPNPNPNPNPNPNPNPNPNPNPNPNPNPNPNPNPNPNPNPNPNP|
#
# |------0.1-----|
# |-Positive(Se)-|

import numpy as np
from PIL import Image
from tqdm import tqdm
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img

path = '/media/kai/4tb/anolis_dataset_for_DL/'

all_data_X_size = 64840
Scl_Sam_rate, Se_rate = 0.9, 0.1 

d_size_Se = int(all_data_X_size * Se_rate)
d_size_Scl_Sam = int(all_data_X_size * Scl_Sam_rate) - d_size_Se #d_size_Se分追加削除

mem_X = np.memmap(path + 'data_X.dat', dtype = 'float16', mode = 'r', shape = (all_data_X_size, 224, 224, 3))
mem_Y = np.memmap(path + 'data_Y.dat', dtype = 'float16', mode = 'r', shape = (all_data_X_size, 1))
mem_img_Y = np.memmap(path + 'data_img_Y.dat', dtype = 'float16', mode = 'r', shape = (int(all_data_X_size / 2), 224, 224, 1))

mem_X_Scl_Sam = np.memmap(path + 'X_Scl_Sam.dat', dtype = 'float16', mode = 'w+', shape = (d_size_Scl_Sam, 224, 224, 3))
mem_Y_Scl_Sam = np.memmap(path + 'Y_Scl_Sam.dat', dtype = 'float16', mode = 'w+', shape = (d_size_Scl_Sam, 1))
mem_X_Se = np.memmap(path + 'X_Se.dat', dtype = 'float16', mode = 'w+', shape = (d_size_Se, 224, 224, 3))
mem_Y_Se = np.memmap(path + 'Y_Se.dat', dtype = 'float16', mode = 'w+', shape = (d_size_Se, 224, 224, 1))

#元データの構成が左半分Positive右半分Negative
all_data_midpoint = int(all_data_X_size / 2)
Scl_Sam_data_midpoint = int(d_size_Scl_Sam / 2)
half_d_size_Scl_Sam = int(d_size_Scl_Sam / 2)

for Scl_Sam_idx in tqdm(range(0, half_d_size_Scl_Sam)):
    mem_X_Scl_Sam[2 * Scl_Sam_idx] = mem_X[Scl_Sam_idx]
    mem_Y_Scl_Sam[2 * Scl_Sam_idx] = mem_Y[Scl_Sam_idx]
    mem_X_Scl_Sam[2 * Scl_Sam_idx + 1] = mem_X[all_data_midpoint + Scl_Sam_idx]
    mem_Y_Scl_Sam[2 * Scl_Sam_idx + 1] = mem_Y[all_data_midpoint + Scl_Sam_idx]

for Se_idx in tqdm(range(0, d_size_Se)):
    mem_X_Se[Se_idx] = mem_X[half_d_size_Scl_Sam + Se_idx]
    mem_Y_Se[Se_idx] = mem_img_Y[half_d_size_Scl_Sam + Se_idx]

mem_X_Scl_Sam.flush()
mem_Y_Scl_Sam.flush()
mem_X_Se.flush()
mem_Y_Se.flush()

save_img('X_Scl_Sam.png', mem_X_Scl_Sam[0])
save_img('X_Se.png', mem_X_Se[0])
save_img('Y_Se.png', mem_Y_Se[0])

print(mem_Y_Scl_Sam[0:10])
