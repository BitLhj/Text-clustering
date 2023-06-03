import os
import numpy as np
import matplotlib.pyplot as plt
import re
from utils import remove_stop_word
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
from random import shuffle
from scipy.sparse import save_npz

train_path = './data/20news-bydate-train'
global_dic = {}
document_words = []

for DirName in os.listdir(train_path):
    DirPath = os.path.join(train_path, DirName)
    for FileName in os.listdir(DirPath):
        FilePath = os.path.join(DirPath, FileName)
        f = open(FilePath, 'r', encoding='utf-8', errors='ignore')
        words = []
        dic = {}
        for line in f:
            line = line.lower()
            words = words + re.findall('[a-zA-Z]+', line)
        for word in words:
            dic[word] = 1 if word not in dic.keys() else dic[word]+1
            global_dic[word] = 1 if word not in global_dic.keys() else global_dic[word] + 1
        # print(dic)
        dic = remove_stop_word(dic)
        words = remove_stop_word(words)
        words = " ".join(words)
        document_words.append(words)


global_dic = remove_stop_word(global_dic)
ls = list(global_dic.items())
ls.sort(key=lambda x:x[1],reverse=True)
print(ls[:50])

vocabulary = []
d_feature = 50
for i in range(d_feature):
    vocabulary.append(ls[i][0])

vectorizer = CountVectorizer(max_features=d_feature,vocabulary=vocabulary)
tf_idf_transformer = TfidfTransformer()

X = tf_idf_transformer.fit_transform(vectorizer.fit_transform(document_words))
# print(X)
# print(vectorizer.get_feature_names)
# print(type(X))
save_npz('x.npz', X, compressed=True)


