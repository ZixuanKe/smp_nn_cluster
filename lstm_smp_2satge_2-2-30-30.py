#coding: utf-8

'''
二阶段法： 先分到对应的聚类中心 后再分小类
'''
# from __future__ import print_function

import gzip
import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.models import load_model
from sklearn.metrics import fbeta_score, precision_recall_fscore_support, \
    roc_curve, auc
from sklearn.metrics import accuracy_score
np.random.seed(1337)    #保持一致性
swda_path = 'swda'
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

print 'Loading data...'
x_train_1, y_train_1, x_test_1, y_test_1 = load_data(max_features,dataset='smp.pretrain.pkl.gz')


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


y_test_1_ = y_test_1


#============================== 阶段1 判断 “定义” 与 “未定义” =============================
# model_1 = load_model('maxf_smp_2train_2develop_valid_weights.epoch:00-val_acc:0.95.hdf5')
#
# predict_class_1 = model_1.predict_classes(x_test_1,batch_size=batch_size)
#
# next_stage = []
#
# for i in range(len(predict_class_1)):
#     if predict_class_1[i] == 0:   #“未定义”类为0
#         continue                  #0即 “未定义类”
#     else:
#         predict_class_1[i] = 1    #“已定义类”放入清1 进入下一轮
#         next_stage.append(i)    #否则对应小标的句子进入下一轮41-41
#
# print 'acc stage_1: '
# print precision_recall_fscore_support(y_test_1, predict_class_1,labels=[0],beta=1)
#
# np.save('next_stage_2_valid.npy',next_stage)
# np.save('predict_class_1_2_valid.npy',predict_class_1)

# 不同结构模型load两个会有偏差报错
# 1. load_weight
# 2. 保存中间结果

#============================== 阶段2 30-30 =============================


next_stage = np.load('next_stage_2_valid.npy')
predict_class_1 = np.load('predict_class_1_2_valid.npy')

model_2 = load_model('maxf_smp_30train_30develop_weights.epoch:500-val_acc:0.74.hdf5') #勿重复load


x_test_2 = x_test_1[next_stage] #幸存者进入第二阶段
print x_test_1.shape
print x_test_2.shape


predict_class_2 = model_2.predict_classes(x_test_2,batch_size=batch_size)

for i in range(len(predict_class_2)):
    if predict_class_2[i] >= 0:
         predict_class_2[i] = predict_class_2[i] + 1  # 30-30 时有减1

result_cla = predict_class_1
num = 0
for i in range(len(result_cla)):
    if result_cla[i] != 0:  #属于第二阶段 直接选用阶段2结果
        result_cla[i] = predict_class_2[num]
        num = num + 1


print accuracy_score(y_test_1_, result_cla)
print precision_recall_fscore_support(y_test_1_, result_cla,labels=[0],beta=1)

