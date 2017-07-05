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

weights_filepath = 'maxf_ch2r_' + str(n_classes) + 'train_' + str(n_classes) + 'split_' + str(s) + 'develop_weights.epoch:{epoch:02d}-val_acc:{val_acc:.2f}.hdf5'

print 'Loading data'
x_train_1, y_train_1, x_test_1, y_test_1= load_data(max_features, dataset='ch2r.pretrain.pkl.gz')



# 直接读取tag才可统一

# x_train_1 = np.load('1_train_test_x_train_kf.npy')
# x_test_1 = np.load('1_train_test_x_test_kf.npy')
# y_train_1 = np.load('1_train_test_y_train_kf.npy')
# y_test_1 = np.load('1_train_test_y_test_kf.npy')




print len(x_test_1), 'test sequences'
print 'Pad sequences (samples x time)'


# Memory 足够时用
x_test_1 = sequence.pad_sequences(x_test_1, maxlen=maxlen)
x_test_1 = x_test_1.reshape((len(x_test_1),maxlen,len_wv))
   # 测试也需要对齐 否则报错： 与模型内输出无关
   # 重复load报错： 与模型内输出无关

#============================== Baseline Model 31-31: 与直接输出对比==========================================
# model_t= load_model('maxf_smp_31train_31develop_weights.epoch:150-val_acc:0.72.hdf5')
# predict_class_t = model_t.predict_classes(x_test_1,batch_size=batch_size)
#
# print accuracy_score(y_test_1, predict_class_t)
# print precision_recall_fscore_support(y_test_1, predict_class_t,labels=[0],beta=1)
#

y_test_1_ = y_test_1


# #============================== 阶段1 判断 “定义” 与 “未定义” =============================
model_1 = load_model('maxf_ch2r_7train_7split_1develop_weights.epoch:154-val_acc:0.80.hdf5')

predict_class_1 = model_1.predict_classes(x_test_1,batch_size=batch_size)

print predict_class_1

next_stage = []

for i in range(len(predict_class_1)):
    if predict_class_1[i] == n:   #“未定义”类为n
        predict_class_1[i] = undefine_seq # 恢复
        continue                  #n即 “未定义类”
    else:                           #“已定义”进入下一轮
        next_stage.append(i)

print 'acc stage_1: '
print precision_recall_fscore_support(y_test_1, predict_class_1,labels=[undefine_seq],beta=1)

np.save('next_stage_7.npy',next_stage)
np.save('predict_class_7.npy',predict_class_1)

# 不同结构模型load两个会有偏差报错
# 1. load_weight
# 2. 保存中间结果
#
# #============================== 阶段2 30-30 =============================
#

next_stage = np.load('next_stage_7.npy')
predict_class_1 = np.load('predict_class_7.npy')

model_2 = load_model('maxf_ch2r_25train_25split_1develop_weights.epoch:206-val_acc:0.54.hdf5') #勿重复load


x_test_2 = x_test_1[next_stage] #幸存者进入第二阶段
print x_test_1.shape
print x_test_2.shape


predict_class_2 = model_2.predict_classes(x_test_2,batch_size=batch_size)

for i in range(len(predict_class_2)):
    if predict_class_2[i] >= undefine_seq:
         predict_class_2[i] = predict_class_2[i] + 1

result_cla = predict_class_1
num = 0
for i in range(len(result_cla)):
    if result_cla[i] != undefine_seq:  #属于第二阶段 直接选用阶段2结果
        result_cla[i] = predict_class_2[num]
        num = num + 1


print accuracy_score(y_test_1_, result_cla)
print precision_recall_fscore_support(y_test_1_, result_cla,labels=[undefine_seq],beta=1)

