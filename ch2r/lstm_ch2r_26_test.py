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
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, \
    roc_curve, auc, accuracy_score

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


max_features = 2436
maxlen = 50 # cut texts after this number of words (among top max_features most common words)
# 限定最大词数

batch_size = 25
len_wv = 50
n_folds = 5
n_classes = 26
s = 1

print 'Loading data'
x_train, y_train, x_test, y_test= load_data(max_features, dataset='ch2r.pretrain.pkl.gz')

# Memory 足够时用
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


x_train = x_train.reshape((len(x_train),maxlen,len_wv))
x_test = x_test.reshape((len(x_test),maxlen,len_wv))


print type(x_train[1][1])
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)



model = load_model('maxf_ch2r_26train_26split_1develop_weights.epoch:609-val_acc:0.57.hdf5')



# val_acc 均 1
# val_loss 也不可


predict_class = model.predict_classes(x_test,batch_size=batch_size)

print predict_class
print y_test
print accuracy_score(y_test, predict_class)
print precision_recall_fscore_support(y_test, predict_class,labels=[11],beta=1)
