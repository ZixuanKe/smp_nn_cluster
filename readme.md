## smp_nn_cluster

---
* cases.xlsx: 分类的cases输出，可以通过筛选得到bad caese
* smp.tags: 分类标签 + 统计 +　标签对应序号
* smp.pretrain.pkl.gz: 原始句子变为向量后的保存结果，后续直接读取即可
* lstm_smp_31train_31test.py: lstm直接进行分类的代码
* smp_topic_cluster_n.py: 聚类代码 n即聚类中心数目

* 待探究： lstm->cnn 