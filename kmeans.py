import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics
from random import shuffle
from scipy.sparse import load_npz

X = load_npz('x.npz')
k = 5
model = KMeans(n_clusters=k, init='k-means++', max_iter=50, n_init=1, verbose=0)

model.fit(X)

X_embedded = TSNE(n_components=2).fit_transform(X)


cb = ['#F0F8FF',
      '#FAEBD7',
      '#00FFFF',
      '#7FFFD4',
      '#F0FFFF',
      '#F5F5DC',
      '#FFE4C4',
      '#000000',
      '#FFEBCD',
      '#0000FF',
      '#8A2BE2',
      '#A52A2A',
      '#DEB887',
      '#5F9EA0',
      '#7FFF00',
      '#D2691E',
      '#FF7F50',
      '#6495ED',
      '#FFF8DC',
      '#DC143C',
      '#00FFFF',
      '#00008B'
      ]
c = []
for i in model.labels_:
    c.append(cb[i])

plt.scatter(X_embedded[:, 0], X_embedded[:, 1], color=c)
pgjg1 = metrics.silhouette_score(X, model.labels_, metric='euclidean')   #轮廓系数
print('聚类结果的轮廓系数=', pgjg1)
try:
      plt.title('K = {}, Silhouette Coefficient = {:.2f}'.format(k,pgjg1))
except:
      pass
plt.show()