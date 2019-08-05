import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, jit, prange

class tools():
    def __init__ (self):
        pass

    @cuda.jit
    def counter_for_IoU(self, attention_map, label_image, result):
        self.bIdx = self.cuda.blockIdx.x #batch
        self.bIdy = self.cuda.blockIdx.y #width
        self.bIdz = self.cuda.blockIdx.z #height
        self.TP, self.FN, self.FP = self.result[1], self.result[2], self.result[3]
        if attention_map[self.bIdx][self.bIdy][self.bIdz] == np.ones((1)):
            if label_image[self.bIdx][self.bIdy][self.bIdz] == onp.ones((1)): 
                self.TP += 1
                self.FP += 1
        if attention_map[self.bIdx][self.bIdy][self.bIdz] == np.zeros((1)):
            if label_image[self.bIdx][self.bIdy][self.bIdz] == np.ones((1)):
                self.FN += 1
        return

    def IoU(self, attention_map, label_image):
        self.shape = label_image.shape
        self.result = np.zeros((1, 1, 1)).astype(np.float32) # TP, FN, FP
        self.each_sum = self.counter_for_IoU[(shape[0], shape[1], shape[2]), 1](attention_map, label_image, self.result)
        return float(self.result[0]) / float(np.sum(self.result))
          
    def get_accuracy_on_batch(self, y_pred, y_label):
        self.Is_equal = tf.equal(tf.to_float(tf.greater(y_pred, 0.5)), tf.to_float(y_label))
        self.accuracy = tf.reduce_mean(tf.cast(self.Is_equal, tf.float32))
        return self.accuracy

    def get_loss_on_batch(self, loss_func): return tf.reduce_mean(loss_func)

    def correct_count_on_batch(self, y_pred, y_label):
        self.Is_equal = tf.equal(tf.to_float(tf.greater(y_pred, 0.5)), tf.to_float(y_label))
        self.correct_sum = tf.reduce_sum(tf.cast(self.Is_equal, tf.float32))
        return self.correct_sum

    #グラフ表示
    def show_figure_of_history(self, loss, acc, val_loss, val_acc):
        self.plot_range = len(loss)
        plt.rc('font', family='serif')
        plt.plot(range(self.plot_range), loss, label = 'loss', color = 'black')
        plt.plot(range(self.plot_range), acc, label='acc',color='blue')
        plt.plot(range(self.plot_range), val_loss, label = 'val_loss', color = 'green')
        plt.plot(range(self.plot_range), val_acc, label = 'val_acc', color = 'red')
        plt.xlabel('epochs')
        #plt.legend()
        #plt.show()
        plt.savefig('figure.pdf')
        return
        
    #history to txt
    def save_history_as_txt(self, loss, acc, val_loss, val_acc, dir_path = ''):
        #self.loss_array = np.array(loss)
        #self.acc_array = np.array(acc)
        #self.val_loss_array = np.array(val_loss)
        #self.val_acc_array = np.array(val_acc)
        np.savetxt(dir_path + 'loss.txt', loss, delimiter = ',')
        np.savetxt(dir_path + 'acc.txt', acc, delimiter = ',')
        np.savetxt(dir_path + 'val_loss.txt', val_loss, delimiter = ',')
        np.savetxt(dir_path + 'val_acc.txt', val_acc, delimiter = ',')
        return

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

