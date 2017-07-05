#coding: utf-8
import  pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import sys
import gzip
import cPickle as pickle
import logging
from gensim.models import Word2Vec
import re
import cPickle as pickle
import gzip
import numpy
from operator import itemgetter
from collections import defaultdict
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

reload(sys)
sys.setdefaultencoding('utf-8')

def save_data(data, pickle_file):
    f = gzip.GzipFile(pickle_file, 'w')
    pickle.dump(data, f)
    f.close()

if __name__ == '__main__':

# ========================================== Ch2r 读取数据方案 ==========================


    data = pd.read_csv( 'ch2r_data.csv',
        sep='\t', encoding='utf8', header=0)


    print('test data shape is :%s'%(str(data.shape)))


    data = data[['LABEL','SENTENCE']]

    data['Words'] = [" ".join(jieba.cut(str(sentence))) for sentence in data['SENTENCE']]

    # ==================================
    # 以文件形式保存下来


    data.to_csv('v2.4_train_2437_word.csv',sep='\t',encoding='utf8')

# 不应分开进行 否则tag序号不一致 # train: [0-2298]
# ==========================================  SWDA 读取数据方案: 读取 W2V 拼接 保存 ==========================


smp_path = 'ch2r'
model_file = 'weibodata_vectorB.gem'
tags_file = 'ch2r.tags'
data_file = 'ch2r.pretrain.pkl.gz'

word_pattern = re.compile(r'[a-z\']+')
except_words = ('and', 'of', 'to')
accept_words = ('i',)

def CorpusReader():
    # 符合SMP的 数据读取方式
        return data

def str2wordlist(s):
    words = [w.split('\'')[0] for w in word_pattern.findall(s)]
    return [w for w in words if (
        len(w) > 1 and w not in except_words or w in accept_words)]


# convert each utterance into a list of word vectors(presented as list),
# convert tag into it's number. return a list with element formed like
# ([word_vec1, word_vec2, ...], tag_no)
def process_data(model, tags):
    x = []
    y = []
    model_cache = {}
    non_modeled = set()
    corpus = CorpusReader()
    for index, utt in corpus.iterrows():
        print utt['Words']
        wordlist = utt['Words']
        for word in wordlist:
            if word in model:
                if word not in model_cache:
                    model_cache[word] = model[word].tolist()
                    print 'dim: ' + str(len(model_cache[word]))  #输出维度 读取时用
                    # 有50维
            else:
                non_modeled.add(word)
        words = [model_cache[w] for w in wordlist if w in model_cache]
        tag = tags[utt['LABEL']] #tag 就是对应 tag_set 里的No.
        x.append(words)
        y.append(tag)
    print 'Complete. The following words are not converted: '
    print list(non_modeled)
    print len(list(non_modeled)) #输出没有被转为词向量的词个数 （删除处理）
    return (x, y)


def save_data(data, pickle_file):
    f = gzip.GzipFile(pickle_file, 'w')
    pickle.dump(data, f)
    f.close()


# load corpus and save number of tags into tags_file.
def preprocess_data():
    act_tags = defaultdict(lambda: 0)
    corpus = CorpusReader()
    for index, utt in corpus.iterrows():
        act_tags[utt['LABEL']] += 1
    act_tags = act_tags.iteritems()
    act_tags = sorted(act_tags, key=itemgetter(1), reverse=True)
    f = open(tags_file, 'w')
    for k, v in act_tags:
        f.write('%s %d\n' % (k, v))
    f.close()   # save tag and its number accordingly
    return dict([(act_tags[i][0], i) for i in xrange(len(act_tags))])


def main():
    print 'Preprocessing data ...'
    tags = preprocess_data()
    print 'Loading model ...'
    # model = KeyedVectors.load_word2vec_format(model_file, binary=True)
    model = Word2Vec.load('weibodata_vectorB.gem')
    #格式暂未知 是否可以设置为别的 / 询问 维度多少？
    print 'Reading and converting data from swda ...'
    data = process_data(model, tags)
    print 'Saving ...'
    save_data(data, data_file)


if __name__ == '__main__':
    main()


# 遍历 pandas
    # for index, row in data.iterrows():   # 获取每行的index、row
    #     for col_name in data.columns:
    #         row[col_name] = exp(row[col_name]) # 把结果返回给data
    # return data
