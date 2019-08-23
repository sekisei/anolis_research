import numpy as np
import glob
from PIL import Image
from tqdm import tqdm

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

def make_dataset(file_list_dict = None, path = None):
    all_data_X_size = len(file_list_dict['x_file_list'])

    mem_X = np.memmap(path + 'data_X.dat', dtype = 'float16', mode = 'w+', shape = (all_data_X_size, 224, 224, 3))
    mem_img_Y = np.memmap(path + 'data_img_Y.dat', dtype = 'float16', mode = 'w+', shape = (int(all_data_X_size / 2), 224, 224, 1))
    mem_Y = np.memmap(path + 'data_Y.dat', dtype = 'float16', mode = 'w+', shape = (all_data_X_size, 1))

    others_class_Label = np.array([0 for index in range(0, int(all_data_X_size / 2))])
    anolis_class_Label = np.array([1 for index in range(0, int(all_data_X_size / 2))])

    for idx, file_name in tqdm(enumerate(file_list_dict['x_file_list'])):
        mem_X[idx] = np.array(Image.open(file_name)).reshape((224, 224, 3))
    mem_X.flush()

    for idx, file_name in tqdm(enumerate(file_list_dict['pos_img_label_file_list'])):
        mem_img_Y[idx] = np.array(Image.open(file_name)).reshape((224, 224, 1))
    mem_img_Y.flush()

    mem_Y[:] = np.hstack((anolis_class_Label, others_class_Label)).reshape((all_data_X_size, 1))[:]
    mem_Y.flush()

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
make_dataset(file_list_dict = file_list_dict, path = save_path)
