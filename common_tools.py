#coding: UTF-8

import matplotlib.pyplot as plt
import numpy as np

class tools():
    def __init__ (self):
        pass

    #グラフ表示
    def show_figure_of_history(self, loss = None, acc = None, iou = None):
        plt.rc('font', family='serif')
        
        fig1 = plt.subplot(3,1,1)
        fig1.set_ylabel('acc')
        if acc != None: fig1.plot(acc, label='acc',color = 'blue')
        fig1.legend()

        fig2 = plt.subplot(3,1,2)
        fig2.set_ylabel('iou')
        if iou != None: fig2.plot(iou, label = 'iou', color = 'green')
        fig2.legend()

        fig3 = plt.subplot(3,1,3)
        fig3.set_xlabel('epochs')
        fig3.set_ylabel('loss')
        if loss != None: fig3.plot(loss, label = 'loss', color = 'black')
        fig3.legend()
        
        plt.savefig('figure.pdf')
        
    #history to txt
    def save_history_as_txt(self, loss = None, acc = None, dir_path = '', numbering = None):
        try:
            np.savetxt('{dir_path}loss{numbering}.txt'.format(dir_path = dir_path, numbering = numbering), loss, delimiter = ',')
            np.savetxt('{dir_path}acc{numbering}.txt'.format(dir_path = dir_path, numbering = numbering), acc, delimiter = ',')
        except:
            pass

if __name__ == '__main__':
    base_path = '/media/kai/4tb/anolis_research_result/20190902_GAIN/gain/'

    with open(base_path + 'loss.txt') as f:
        loss_data = f.readlines()
        loss_data = [data.replace('\n', '') for data in loss_data]
        loss_data = list(map(float, loss_data))
    with open(base_path + 'acc.txt') as f:
        acc_data = f.readlines()
        acc_data = [data.replace('\n', '') for data in acc_data]
        acc_data = list(map(float, acc_data))
    with open(base_path + 'iou_acc.txt') as f:
        iou_data = f.readlines()
        iou_data = [data.replace('\n', '') for data in iou_data]
        iou_data = list(map(float, iou_data))

    tool = tools()
    tool.show_figure_of_history(loss = loss_data, acc = acc_data, iou = iou_data)
