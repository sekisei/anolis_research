import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img

base_path = '/media/kai/4tb/'
save_path = '/media/kai/4tb/anolis_dataset_for_DL/'

targets_test_base_path = base_path + 'anolis4/dataset/targets/test/'
#targets_test_base_path = base_path + 'anolis4/dataset/no_augment/target/'
targets_valid_base_path = base_path + 'anolis4/dataset/targets/valid/'
targets_train_base_path = base_path + 'anolis4/dataset/targets/train/'

others_test_base_path = base_path + 'anolis4/dataset/others/test/'
#others_test_base_path = base_path + 'anolis4/dataset/no_augment/others/'
others_valid_base_path = base_path + 'anolis4/dataset/others/valid/'
others_train_base_path = base_path + 'anolis4/dataset/others/train/'

def get_positive_file_list(path = None):
    Label_file_list = []
    Input_file_list = []
    dir_list = glob.glob(path + '*')

    #anolis and image label
    for dir_path in tqdm(dir_list):
        try:
            Label_list = glob.glob(dir_path + '/Label/*')
            Input_list = [file_name for file_name in map(lambda file_name: file_name.replace('Label', 'Input').replace('L', ''), Label_list)]
            Label_file_list.extend(Label_list)
            Input_file_list.extend(Input_list)
        except:
            print('error')
    
    return {'Input_file_list': Input_file_list, 'Label_file_list': Label_file_list}

def get_negative_file_list(path = None):
    file_list = glob.glob(path + '*')
    print(len(file_list))
    return file_list

def make_X(file_list = None, save_path = None):
    all_data_size = len(file_list[0])
    mem_X = np.memmap(save_path + 'data_X.dat', dtype = 'float16', mode = 'w+', shape = (all_data_size, 224, 224, 3))
    for idx, (pos_file_name, neg_file_name) in enumerate(tqdm(zip(file_list[0], file_list[1]))):
        mem_X[2 * idx] = np.array(Image.open(pos_file_name)).reshape((224, 224, 3))
        mem_X[2 * idx + 1] = np.array(Image.open(neg_file_name)).reshape((224, 224, 3))
    mem_X.flush()

def make_Y(data_size = None, save_path = None):
    mem_Y = np.memmap(save_path + 'data_Y.dat', dtype = 'float16', mode = 'w+', shape = (data_size, 1))    
    for idx in range(0, int(data_size / 2)):
        mem_Y[2 * idx] = 1
        mem_Y[2 * idx + 1] = 0
    mem_Y.flush()

def make_Y_img(file_list = None, save_path = None):
    data_size = len(file_list) * 2
    mem_Y_img = np.memmap(save_path + 'data_Y_img.dat', dtype = 'float16', mode = 'w+', shape = (data_size, 224, 224, 1))
    for idx, file_name in enumerate(tqdm(file_list)):
        mem_Y_img[2 * idx] = np.array(Image.open(file_name)).reshape((224, 224, 1))
        mem_Y_img[2 * idx + 1] = np.zeros((224, 224, 1))
    mem_Y_img.flush()

pos_test_file_list = get_positive_file_list(path = targets_test_base_path)
pos_valid_file_list = get_positive_file_list(path = targets_valid_base_path)
pos_train_file_list = get_positive_file_list(path = targets_train_base_path)
neg_test_file_list = get_negative_file_list(path = others_test_base_path)
neg_valid_file_list = get_negative_file_list(path = others_valid_base_path)
neg_train_file_list = get_negative_file_list(path = others_train_base_path)

all_pos_X_file_list = []
all_neg_X_file_list = []
all_Yimg_file_list = []
all_pos_X_file_list.extend(pos_test_file_list['Input_file_list'])
all_pos_X_file_list.extend(pos_valid_file_list['Input_file_list'])
all_pos_X_file_list.extend(pos_train_file_list['Input_file_list'])
all_neg_X_file_list.extend(neg_test_file_list)
all_neg_X_file_list.extend(neg_valid_file_list)
all_neg_X_file_list.extend(neg_train_file_list)
all_Yimg_file_list.extend(pos_test_file_list['Label_file_list'])
all_Yimg_file_list.extend(pos_valid_file_list['Label_file_list'])
all_Yimg_file_list.extend(pos_train_file_list['Label_file_list'])

all_data_size = len(all_pos_X_file_list) + len(all_neg_X_file_list)
mem_X = np.memmap(save_path + 'data_X.dat', dtype = 'float16', mode = 'w+', shape = (all_data_size, 224, 224, 3))
mem_Y = np.memmap(save_path + 'data_Y.dat', dtype = 'float16', mode = 'w+', shape = (all_data_size, 1))
mem_Y_img = np.memmap(save_path + 'data_Y_img.dat', dtype = 'float16', mode = 'w+', shape = (all_data_size, 224, 224, 1))

for idx, (pos_file_name, neg_file_name, img_label_file_name) in enumerate(tqdm(zip(all_pos_X_file_list, all_neg_X_file_list, all_Yimg_file_list))):
    mem_X[2 * idx] = np.array(Image.open(pos_file_name)).reshape((224, 224, 3))
    mem_X[2 * idx + 1] = np.array(Image.open(neg_file_name)).reshape((224, 224, 3))
    mem_Y[2 * idx] = 1
    mem_Y[2 * idx + 1] = 0
    mem_Y_img[2 * idx] = np.array(Image.open(img_label_file_name)).reshape((224, 224, 1))
    mem_Y_img[2 * idx + 1] = np.zeros((224, 224, 1))

mem_X.flush()
mem_Y.flush()
mem_Y_img.flush()

print(mem_X.shape)
print(mem_Y.shape)
print(mem_Y_img.shape)

#save_img('1.png', mem_X[1])
#save_img('1_Y.png', mem_Y_img[1])
#print(mem_Y[1])
    
'''
def make_dataset(file_list_dict = None, path = None):
    all_data_X_size = len(file_list_dict['x_file_list'])

    mem_X = np.memmap(path + 'data_X.dat', dtype = 'float16', mode = 'w+', shape = (all_data_X_size, 224, 224, 3))
    mem_img_Y = np.memmap(path + 'data_img_Y.dat', dtype = 'float16', mode = 'w+', shape = (int(all_data_X_size / 2), 224, 224, 1))
    mem_Y = np.memmap(path + 'data_Y.dat', dtype = 'float16', mode = 'w+', shape = (all_data_X_size, 1))

    others_class_Label = np.array([0 for index in range(0, int(all_data_X_size / 2))])
    anolis_class_Label = np.array([1 for index in range(0, int(all_data_X_size / 2))])

    for idx, file_name in enumerate(tqdm(file_list_dict['x_file_list'])):
        mem_X[idx] = np.array(Image.open(file_name)).reshape((224, 224, 3))
    mem_X.flush()

    for idx, file_name in enumerate(tqdm(file_list_dict['pos_img_label_file_list'])):
        mem_img_Y[idx] = np.array(Image.open(file_name)).reshape((224, 224, 1))
    mem_img_Y.flush()

    mem_Y[:] = np.hstack((anolis_class_Label, others_class_Label)).reshape((all_data_X_size, 1))[:]
    mem_Y.flush()
    
def make_dataset_with_mask(file_list_dict = None, path = None):
    all_data_X_size = len(file_list_dict['x_file_list'])

    #マスク付きテスト用画像作成 (トカゲのみ対象の処理)
    def img_masking(imgs = None):
        (X, Y_img) = (imgs[0], imgs[1])
        #画素値の反転
        Y_img_reverse = np.abs(Y_img - Y_img.max())
        Y_img_reverse = Y_img_reverse / Y_img_reverse.max()
        X_masking = (Y_img_reverse * X).astype(np.int)
        return X_masking  

    mem_X = np.memmap(path + 'data_X.dat', dtype = 'float16', mode = 'r', shape = (all_data_X_size, 224, 224, 3))
    mem_X_mask = np.memmap(path + 'data_X_masking.dat', dtype = 'float16', mode = 'w+', shape = (all_data_X_size, 224, 224, 3))
    mem_img_Y = np.memmap(path + 'data_img_Y.dat', dtype = 'float16', mode = 'r', shape = (int(all_data_X_size / 2), 224, 224, 1))

    #マスク付き画像の保存
    for idx, file_name in enumerate(tqdm(file_list_dict['x_file_list'])):
        mem_X_mask[idx] = np.array(Image.open(file_name)).reshape((224, 224, 3))
    mem_X_mask.flush()
    
    for idx in tqdm(range(0, int(all_data_X_size / 2))):
        X_mask = img_masking(imgs = (mem_X[idx], mem_img_Y[idx]))
        mem_X_mask[idx] = X_mask

    mem_X_mask.flush()

    #save_img('0.png', mem_X_mask[0])
    #save_img('32419.png', mem_X_mask[32419])
    #save_img('32420.png', mem_X_mask[32420])

pos_test_file_list = get_positive_file_list(path = targets_test_base_path)
pos_valid_file_list = get_positive_file_list(path = targets_valid_base_path)
pos_train_file_list = get_positive_file_list(path = targets_train_base_path)
neg_test_file_list = get_negative_file_list(path = others_test_base_path)
neg_valid_file_list = get_negative_file_list(path = others_valid_base_path)
neg_train_file_list = get_negative_file_list(path = others_train_base_path)
    
pos_test_file_list['Input_file_list'].extend(pos_valid_file_list['Input_file_list'])
pos_test_file_list['Input_file_list'].extend(pos_train_file_list['Input_file_list'])
neg_test_file_list.extend(neg_valid_file_list)
neg_test_file_list.extend(neg_train_file_list)
pos_all_x_list = pos_test_file_list['Input_file_list'][:]
neg_all_x_list = neg_test_file_list[:]
pos_all_x_list.extend(neg_all_x_list)
x_file_list = pos_all_x_list

pos_test_file_list['Label_file_list'].extend(pos_valid_file_list['Label_file_list'])
pos_test_file_list['Label_file_list'].extend(pos_train_file_list['Label_file_list'])
pos_img_label_file_list = pos_test_file_list['Label_file_list'][:]

file_list_dict = {
    'x_file_list': x_file_list,
    'pos_img_label_file_list': pos_img_label_file_list,
}

#print(pos_file_list)
#make_dataset(file_list_dict = file_list_dict, path = save_path)
#make_dataset_with_mask(file_list_dict = file_list_dict, path = save_path)
'''
