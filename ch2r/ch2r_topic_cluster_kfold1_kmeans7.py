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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals import joblib
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
n_classes = 25
s = 1
undefine_seq = 11
n= 7 # 聚类中心数

print 'Loading data'
# x_train, y_train, x_test, y_test= load_data(max_features, dataset='ch2r.pretrain.pkl.gz')



# 直接读取tag才可统一

x_train = np.load('1_train_test_x_train_kf.npy')
x_test = np.load('1_train_test_x_test_kf.npy')
y_train = np.load('1_train_test_y_train_kf.npy')
y_test = np.load('1_train_test_y_test_kf.npy')




# ======================== 删除 所有chat类（“其他类”/“未定义类”） ==============================

#train
delete_no = []
for i in range(len(y_train)):
    if y_train[i] == undefine_seq:  #chat 为 11
        delete_no.append(i)

x_train = np.delete(x_train,delete_no,axis=0)
y_train = np.delete(y_train,delete_no,axis=0)

#test
delete_no = []
for i in range(len(y_test)):
    if y_test[i] == undefine_seq:   #chat 为 11
        delete_no.append(i)

x_test = np.delete(x_test,delete_no,axis=0)
y_test = np.delete(y_test,delete_no,axis=0)


# 对齐
for i in range(len(y_train)):
    if y_train[i] > undefine_seq:
        y_train[i] = y_train[i]-1   #  Tag需要暂时改变

for i in range(len(y_test)):
    if y_test[i] > undefine_seq:
        y_test[i] = y_test[i] - 1    #  Tag需要暂时改变

# ======================== 删除 所有chat类（“其他类”/“未定义类”）， Tag改变 ==============================


# y_train = np.array(y_train)
# x_train = np.array(x_train)

print len(x_train), 'train sequences'
print len(x_test), 'test sequences'
print x_train.shape
print y_train.shape

# Memory 足够时用
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


x_train = x_train.reshape((len(x_train),maxlen*len_wv))
x_test = x_test.reshape((len(x_test),maxlen*len_wv))


print type(x_train[1][1])
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# ============================================以上获取数据=====================================
# k_ID = 2/3/4/5/6/7 粒度加大
# k_OOD = 2/3/4/5/6/7 粒度减小


kmeans =  MiniBatchKMeans(max_iter=1000,n_clusters=n)
kmeans = kmeans.fit(x_train)

joblib.dump(kmeans , 'minikmeans_' + str(n) + '_ch2r_kfold_1.pkl')


print kmeans.labels_






