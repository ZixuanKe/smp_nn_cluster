#coding: utf-8


import numpy as np, matplotlib.pyplot as plt
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.datasets.base import load_iris
from sklearn.manifold.t_sne import TSNE
from sklearn.linear_model.logistic import LogisticRegression
from keras.models import load_model
import pickle
import gzip
from keras.preprocessing import sequence
import seaborn as sns


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

print 'Loading data'
x_train, y_train, x_test, y_test= load_data(max_features, dataset='smp.pretrain.pkl.gz')
# 直接读取tag才可统一

# replace the below by your data and model
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
x_test = x_test.reshape((len(x_test),maxlen*len_wv))


print len(x_train), 'train sequences'
print len(x_test), 'test sequences'




X_Train_embedded = TSNE(n_components=2).fit_transform(x_test)
print X_Train_embedded.shape


x_test = x_test.reshape((len(x_test),maxlen,len_wv))

model= load_model('lstm_smp_31train_31develop_paraset_1_split_0_weights.epoch:316-val_acc:0.78.hdf5')
y_predicted = model.predict_classes(x_test,batch_size=batch_size)
# replace the above by your data and model

# create meshgrid
resolution = 100 # 100x100 background pixels
X2d_xmin, X2d_xmax = np.min(X_Train_embedded[:,0]), np.max(X_Train_embedded[:,0])
X2d_ymin, X2d_ymax = np.min(X_Train_embedded[:,1]), np.max(X_Train_embedded[:,1])
xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution), np.linspace(X2d_ymin, X2d_ymax, resolution))

# approximate Voronoi tesselation on resolution x resolution grid using 1-NN
background_model = KNeighborsClassifier(n_neighbors=2).fit(X_Train_embedded, y_predicted)
voronoiBackground = background_model.predict(np.c_[xx.ravel(), yy.ravel()])
voronoiBackground = voronoiBackground.reshape((resolution, resolution))

np.save('xx_knn_31.npy',xx)
np.save('yy_knn_31.npy',yy)
np.save('bg_knn_31.npy',voronoiBackground)
np.save('X_Train_embedded_knn_31.npy',X_Train_embedded)
np.save('y_test_knn_31.npy',y_test)
np.save('predict_class_knn_31.npy',y_predicted)



xx = np.load('xx_knn_31.npy')
yy = np.load('yy_knn_31.npy')
voronoiBackground = np.load('bg_knn_31.npy')
X_Train_embedded = np.load('X_Train_embedded_knn_31.npy')
y_test = np.load('y_test_knn_31.npy')
predict_class = np.load('predict_class_knn_31.npy')

y_test_1 = y_test
# y_test_2 = y_test

# ============ 应该是0 而没有被分到0 ============
for i in range(len(y_test)):
    if y_test[i] == 0:
        if y_test[i] != predict_class[i]:
            y_test_1[i] = 31    # 分错
        else:
            y_test_1[i] = 32    # 分对
# # ============ 应该非0 而被分到0 ============
# for i in range(len(y_test)):
#     if y_test[i] == 0:
#         if y_test[i] != predict_class[i]:
#             y_test_2[i] = 32



from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体

mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
palette = np.array(sns.color_palette("hls", 31))
# plt.contour(xx, yy, voronoiBackground,colors = 'black')
ax = plt.subplot(111)


ax.contourf(xx, yy, voronoiBackground, cmap=plt.cm.Spectral)  #等高线
#
ax.scatter(X_Train_embedded[:, 0][y_test_1==31], X_Train_embedded[:, 1][y_test_1==31],
            marker='^',color='black', alpha=0.7,
            label=u'被错误分类的“未定义”话语')

ax.scatter(X_Train_embedded[:, 0][y_test_1==32], X_Train_embedded[:, 1][y_test_1==32],
            marker='^',color='black', alpha=0.7,facecolors='none',linewidths =1.5,
            label=u'被正确分类的“未定义”话语')

# plt.legend(loc='lower right')
# ,facecolors='none'
# linewidths =2,


# 367 -> 197 老马的电话是多少
# 117->112 你生日什么时候

# ax.text(X_Train_embedded[:, 0][197], X_Train_embedded[:, 1][197],u'老马的电话是多少')
# # ax.text(X_Train_embedded[:, 0][222], X_Train_embedded[:, 1][222],u'怎么煮方便面')
# ax.text(X_Train_embedded[:, 0][112]-2, X_Train_embedded[:, 1][112]-2,u'你生日什么时候')
#

# ax.scatter(X_Train_embedded[:, 0][197], X_Train_embedded[:, 1][197],
#             marker='o',color='black', alpha=0.7,
#             label=u'被错误分类的“已定义”话语')
# # ax.scatter(X_Train_embedded[:, 0][222], X_Train_embedded[:, 1][222],
# #             marker='o',color='black',  alpha=0.7,
# #             label=u'怎么煮方便面')
# ax.scatter(X_Train_embedded[:, 0][112]-2, X_Train_embedded[:, 1][112]-2,
#             marker='^',color='black', alpha=0.7,
#             label=u'被错误分类的“未定义”话语')

# plt.scatter(X_Train_embedded[:, 0][222], X_Train_embedded[:, 1][222],
#             marker='o',color='black', alpha=0.7,
#             label=u'怎么煮方便面')
# plt.scatter(X_Train_embedded[:, 0][117], X_Train_embedded[:, 1][117],
#             marker='^',color='black', alpha=0.7,
#             label=u'怎么你有房有车吗？微笑')
# plt.legend(loc='lower right')

# plt.scatter(X_Train_embedded[:,0][y_test==0],
#             X_Train_embedded[:,1][y_test==0],
#             c=palette[y_test.astype(np.int)])

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])

# Put a legend below current axis
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)


plt.title("LSTM")
plt.show()