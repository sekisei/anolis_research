import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, jit, prange

@cuda.jit
def counter_for_IoU(attention_map, label_image, result):
    bIdx = cuda.blockIdx.x #batch
    bIdy = cuda.blockIdx.y #width
    bIdz = cuda.blockIdx.z #height
    TP, FN, FP = result[1], result[2], result[3]
    if attention_map[bIdx][bIdy][bIdz] == np.ones((1)):
       if label_image[bIdx][bIdy][bIdz] == onp.ones((1)): 
          TP += 1
       FP += 1
    if attention_map[bIdx][bIdy][bIdz] == np.zeros((1)):
       if label_image[bIdx][bIdy][bIdz] == np.ones((1)):
          FN += 1

def IoU(attention_map, label_image):
    shape = label_image.shape
    result = np.zeros((1, 1, 1)).astype(np.float32) # TP, FN, FP
    each_sum = counter_for_IoU[(shape[0], shape[1], shape[2]), 1](attention_map, label_image, result)
    return float(result[0]) / float(np.sum(result))
          
def get_accuracy_on_batch(y_pred, y_label):
    Is_equal = tf.equal(tf.to_float(tf.greater(y_pred, 0.5)), tf.to_float(y_label))
    accuracy = tf.reduce_mean(tf.cast(Is_equal, tf.float32))
    return accuracy

def get_loss_on_batch(loss_func): return tf.reduce_mean(loss_func)

def correct_count_on_batch(y_pred, y_label):
    Is_equal = tf.equal(tf.to_float(tf.greater(y_pred, 0.5)), tf.to_float(y_label))
    correct_sum = tf.reduce_sum(tf.cast(Is_equal, tf.float32))
    return correct_sum

#------------------------------------------------------------------------------------------------------------------------------------------------
#グラフ表示
#------------------------------------------------------------------------------------------------------------------------------------------------
def show_figure_of_history(loss, acc, val_loss, val_acc):
    plot_range = len(loss)
    plt.rc('font',family='serif')
    plt.plot(range(plot_range),loss,label='loss',color='black')
    plt.plot(range(plot_range),acc,label='acc',color='blue')
    plt.plot(range(plot_range),val_loss,label='val_loss',color='green')
    plt.plot(range(plot_range),val_acc,label='val_acc',color='red')
    plt.xlabel('epochs')
    #plt.legend()
    #plt.show()
    plt.savefig('figure.pdf')
#------------------------------------------------------------------------------------------------------------------------------------------------
#history to txt
#------------------------------------------------------------------------------------------------------------------------------------------------
def save_history_as_txt(loss, acc, val_loss, val_acc, dir_path = ''):
    #loss_array=np.array(loss)
    #acc_array=np.array(acc)
    #val_loss_array=np.array(val_loss)
    #val_acc_array=np.array(val_acc)
    np.savetxt(dir_path + 'loss.txt',loss,delimiter=',')
    np.savetxt(dir_path + 'acc.txt',acc,delimiter=',')
    np.savetxt(dir_path + 'val_loss.txt',val_loss,delimiter=',')
    np.savetxt(dir_path + 'val_acc.txt',val_acc,delimiter=',')
#------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    #0.7: 32, 0: 0, 32*2
    pred = [0.2 for i in range(0, 32)]
    label = [1.0 for i in range(0, 32)]
    y_pred = tf.constant([pred, pred])
    y_label = tf.constant([label, label])
    
    Is_equal = tf.equal(tf.to_float(tf.greater(y_pred, 0.5)), y_label)
    accuracy = tf.reduce_mean(tf.cast(Is_equal, tf.float32))
    
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    #sess.run(result)
    #print(accuracy.eval(session=sess))

