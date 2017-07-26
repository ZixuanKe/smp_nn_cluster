#coding: utf-8

'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.
- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
# from __future__ import print_function
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Activation
from keras.layers import LSTM
import gzip
import pickle
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import numpy as np
from keras import regularizers
from sklearn.model_selection import KFold

np.random.seed(1337)    #保持一致性

# word embeding 已经得到 (暂时无法得到Google news)
def load_data(max_features,dataset):
    '''
    读取数据，暂不设验证集
    :param dataset:
    :return:
    '''
    TRAIN_SET = max_features
    f = gzip.open(dataset, 'rb')
    data = pickle.load(f)
    f.close()
    data_x, data_y = data
    train_x = data_x[:TRAIN_SET]
    train_y = data_y[:TRAIN_SET]
    test_x = data_x[TRAIN_SET:]
    test_y = data_y[TRAIN_SET:]
    return train_x, train_y, test_x, test_y


max_features = 2298
maxlen = 50 # cut texts after this number of words (among top max_features most common words)
# 限定最大词数

batch_size = 50
len_wv = 50

n_folds = 5
n_classes = 31
weights_filepath = 'maxf_smp_' + str(n_classes) + 'train_' + str(n_classes) + 'develop_weights.epoch:{epoch:02d}-val_acc:{val_acc:.2f}.hdf5'

print 'Loading data'
x_train, y_train, x_test, y_test= load_data(max_features, dataset='smp.pretrain.pkl.gz')



y_train = np.array(y_train)
x_train = np.array(x_train)

print len(x_train), 'train sequences'
print len(x_test), 'test sequences'
print x_train.shape
print y_train.shape
#



train_valid_split = []

for i in range(n_classes): #共26个类
    split = []

    X = x_train[y_train == i]   #np专用
    y = y_train[y_train == i]

    print 'y'
    print y
    #未彻底完成 仍有错误

    if len(y) <= 5: # 样本数量少
        split.append([X, X, y, y])
        train_valid_split.append(split)
        continue

    kf = KFold(n_splits=n_folds)
    kf.get_n_splits(X)

    for train_index, test_index in kf.split(X,y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        x_train_kf, x_test_kf = X[train_index], X[test_index]
        y_train_kf, y_test_kf = y[train_index], y[test_index]      #一个类中的一份
        split.append([x_train_kf, x_test_kf, y_train_kf, y_test_kf])

        print 'test_index'
        print test_index
        print 'y_test_kf'
        print y_test_kf
    train_valid_split.append(split) # n个类每类folds份

#合并为5个需要进行的实验数据


for i in range(n_folds):
    split_0 = []
    split_1 = []
    split_2 = []
    split_3 = []
    for c in range(n_classes):
            split_0 = np.concatenate((train_valid_split[c][0][0],split_0),axis=0)
            split_1 = np.concatenate((train_valid_split[c][0][1],split_1),axis=0)
            split_2  =  np.concatenate((train_valid_split[c][0][2],split_2),axis=0)
            split_3 = np.concatenate((train_valid_split[c][0][3],split_3),axis=0)

    print 'split_0 ' + str(len(split_0))
    print 'split_1 ' + str(len(split_1))
    print 'split_2 ' + str(len(split_2))
    print 'split_3 ' + str(len(split_3))


    np.save(str(i) + '_train_test_x_train_kf_smp.npy',split_0)
    np.save(str(i) + '_train_test_x_test_kf_smp.npy',split_1)
    np.save(str(i) + '_train_test_y_train_kf_smp.npy',split_2)
    np.save(str(i) + '_train_test_y_test_kf_smp.npy',split_3)

    print split_3

# x_train_kf, x_test_kf, y_train_kf, y_test_kf]


# #保存分得结果

