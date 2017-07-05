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
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from sklearn.metrics import accuracy_score
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
n= 6 # 聚类中心数

kmeans_model = 'minikmeans_' + str(n) + '_ch2r_kfold_1.pkl'
weights_filepath = 'maxf_ch2r_' + str(n) + 'train_' + str(n) + 'split_' + str(s) + 'develop_weights.epoch:{epoch:02d}-val_acc:{val_acc:.2f}.hdf5'

print 'Loading data'
# x_train, y_train, x_test, y_test= load_data(max_features, dataset='ch2r.pretrain.pkl.gz')



# 直接读取tag才可统一

x_train = np.load('1_train_test_x_train_kf.npy')
x_test = np.load('1_train_test_x_test_kf.npy')
y_train = np.load('1_train_test_y_train_kf.npy')
y_test = np.load('1_train_test_y_test_kf.npy')



print len(x_train), 'train sequences'
print len(x_test), 'test sequences'



# Memory 足够时用
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


y_train = np.array(y_train)
y_test = np.array(y_test)

x_new_train = x_train[y_train!=undefine_seq]
x_new_test = x_test[y_test!=undefine_seq]


x_train = x_train[y_train==undefine_seq]
x_test = x_test[y_test==undefine_seq] # 已定义类的直接在后续重新洗牌

y_train = y_train[y_train==undefine_seq]
y_test = y_test[y_test==undefine_seq] # 已定义类的直接在后续重新洗牌

x_train = x_train.reshape((len(x_train),maxlen*len_wv))
x_test = x_test.reshape((len(x_test),maxlen*len_wv))

x_new_train = x_new_train.reshape((len(x_new_train),maxlen*len_wv))
x_new_test = x_new_test.reshape((len(x_new_test),maxlen*len_wv))  #方便结合


# ==================test + cluster =========================
# 写上正确标签 但新测试来的时候 由LSTM去判断
kmeans = joblib.load(kmeans_model)    #仍然用Kmeans_train, 聚类中心由训练集得出
y_new_test = kmeans.predict(x_new_test)

for i in range(len(y_test)):
    y_test[i] = n# 只对已定义类进行分类 0-4 "未定义"类为5

x_test = np.concatenate((x_test,x_new_test),axis=0)
y_test = np.concatenate((y_test,y_new_test),axis=0) #拼接为新类

# ==================train + cluster =========================

kmeans = joblib.load(kmeans_model)
y_new_train = kmeans.predict(x_new_train)

for i in range(len(y_train)):
    y_train[i] = n # 只对已定义类进行分类 "未定义"类为5

x_train = np.concatenate((x_train,x_new_train),axis=0)
y_train = np.concatenate((y_train,y_new_train),axis=0) #拼接为新类


print y_test
print y_train

x_train = x_train.reshape((len(x_train),maxlen,len_wv))
x_test = x_test.reshape((len(x_test),maxlen,len_wv))

print type(x_train[1][1])
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)



# step 1: train / test 均2分类
y_test = np_utils.to_categorical(y_test, n+1) # 必须使用固定格式表示标签
y_train = np_utils.to_categorical(y_train, n+1) # 必须使用固定格式表示标签 一共 42分类


print'Build model...'

# model_b = load_model('maxf_smp_5train_5develop_weights.epoch:44-val_acc:0.95.hdf5')


model = Sequential()

model.add(LSTM(150,return_sequences=True,input_shape=(maxlen,len_wv)))  #维度不同则要保证test也有同样的维度
model.add(LSTM(150,return_sequences=True))

model.add(LSTM(150,return_sequences=True))
model.add(LSTM(150,return_sequences=True))

model.add(LSTM(150,return_sequences=True))
model.add(LSTM(150,return_sequences=True))

model.add(LSTM(150,return_sequences=True))
model.add(LSTM(150,return_sequences=True))

model.add(LSTM(150,return_sequences=True))
model.add(LSTM(150))        #输出不同相差10% 调参影响巨大

model.add(Dense(n+1))
model.add(Activation('softmax'))
#
# for i in range(11):
#     print(i)
#     weights=model_b.layers[i].get_weights()   #获得已经训练好的参数
#     model.layers[i].set_weights(weights)


model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])
print model.metrics_names
print'Train...'
model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test,y_test),
          epochs=1000,shuffle=True,
          callbacks=
          [ModelCheckpoint(weights_filepath,monitor='val_acc',
                           verbose=1, save_best_only=True, mode='max')])
# 如果准确率有提升/epoch 则保存文件
# 不callback 则验证集只能看 不能保存结果





