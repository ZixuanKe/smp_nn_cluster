#coding: utf-8

'''
训练LSTM
所有“已定义”为一类
所有“未定义”为一类
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
from sklearn.metrics import fbeta_score
from keras import backend as K
import numpy as np
from keras import regularizers
from sklearn.externals import joblib

np.random.seed(1337)    #保持一致性

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

weights_filepath = 'maxf_smp_2train_2develop_valid_weights.epoch:{epoch:02d}-val_acc:{val_acc:.2f}.hdf5'

print 'Loading data'
x_train, y_train, x_test, y_test = load_data(max_features,dataset='smp.pretrain.pkl.gz')



print len(x_train), 'train sequences'
print len(x_test), 'test sequences'






# Memory 足够时用
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


y_train = np.array(y_train)
y_test = np.array(y_test)



for i in range(len(y_train)):
    if y_train[i] == 0:
       y_train[i] = 0   # "未定义类"保持0不变
    else:
       y_train[i] = 1   # "已定义类"统一变为1

for i in range(len(y_test)):
    if y_test[i] == 0:
        y_test[i] = 0
    else:
        y_test[i] = 1



x_train = x_train.reshape((len(x_train),maxlen,len_wv))
x_test = x_test.reshape((len(x_test),maxlen,len_wv))

print type(x_train[1][1])
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


# step 1: train / test 均2分类
y_test = np_utils.to_categorical(y_test, 2) # 必须使用固定格式表示标签
y_train = np_utils.to_categorical(y_train, 2) # 必须使用固定格式表示标签 一共 42分类


print'Build model...'

# model_b = load_model('maxf_smp_6train_6develop_weights.epoch:34-val_acc:0.74.hdf5')


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

model.add(Dense(2))
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
model.fit(x_train, y_train, batch_size=batch_size, validation_split=0.3333,
          epochs=1000,shuffle=True,
          callbacks=
          [ModelCheckpoint(weights_filepath,monitor='val_acc',
                           verbose=1, save_best_only=True, mode='max')])
# 如果准确率有提升/epoch 则保存文件
# 不callback 则验证集只能看 不能保存结果



