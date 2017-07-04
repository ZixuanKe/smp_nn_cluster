#coding: utf-8

'''
直接使用LSTM进行31类别的分类
参数：
	10层
	cell: 150个
	句子最大长度： 50词
	每个词向量维度： 50

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

batch_size = 25
len_wv = 50

weights_filepath = 'maxf_smp_31train_31develop_weights.epoch:{epoch:02d}-val_acc:{val_acc:.2f}.hdf5'

print 'Loading data'
x_train, y_train, x_test, y_test= load_data(max_features, dataset='smp.pretrain.pkl.gz')
# 直接读取tag才可统一

print len(x_train), 'train sequences'
print len(x_test), 'test sequences'

# Memory 足够时用
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)


x_train = x_train.reshape((len(x_train),maxlen,len_wv))
x_test = x_test.reshape((len(x_test),maxlen,len_wv))


print type(x_train[1][1])
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)



# step 1: train / test 均2分类
y_test = np_utils.to_categorical(y_test, 31) # 必须使用固定格式表示标签
y_train = np_utils.to_categorical(y_train, 31) # 必须使用固定格式表示标签 一共 42分类



# =================================以上数据读取============================



print'Build model...'

# model_b = load_model('maxf_smp_31train_31develop_weights.epoch:20-val_acc:0.57.hdf5')
#读取已经训练好的模型

model = Sequential()

# Stacked LSTM
model.add(LSTM(150,return_sequences=True,input_shape=(maxlen,len_wv)))
model.add(LSTM(150,return_sequences=True))

model.add(LSTM(150,return_sequences=True))
model.add(LSTM(150,return_sequences=True))

model.add(LSTM(150,return_sequences=True))
model.add(LSTM(150,return_sequences=True))

model.add(LSTM(150,return_sequences=True))
model.add(LSTM(150,return_sequences=True))

model.add(LSTM(150,return_sequences=True))
model.add(LSTM(150))



model.add(Dense(31))
model.add(Activation('softmax'))

# for i in range(11):
#     print(i)
#     weights=model_b.layers[i].get_weights()   #获得已经训练好的参数
#     model.layers[i].set_weights(weights)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

print'Train...'
model.fit(x_train, y_train, batch_size=batch_size, 

			validation_data=(x_test,y_test),	#验证集可更改
			
          epochs=100,shuffle=True,
          callbacks=
          [ModelCheckpoint(weights_filepath,monitor='val_acc',
                           verbose=1, save_best_only=True, mode='max')])
# 如果准确率有提升/epoch 则保存文件
# 不callback 则验证集只能看 不能保存结果



