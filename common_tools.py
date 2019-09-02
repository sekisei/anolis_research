import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numba import cuda, jit, prange

class tools():
    def __init__ (self):
        pass

    #グラフ表示
    def show_figure_of_history(self, loss = None, acc = None, val_loss = None, val_acc = None):
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
    def save_history_as_txt(self, loss = None, acc = None, iou_acc = None, dir_path = ''):
        if loss != None: np.savetxt(dir_path + 'loss.txt', loss, delimiter = ',')
        if acc != None: np.savetxt(dir_path + 'acc.txt', acc, delimiter = ',')
        if iou_acc != None: np.savetxt(dir_path + 'iou_acc.txt', iou_acc, delimiter = ',')
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

