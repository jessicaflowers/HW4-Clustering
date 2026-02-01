import numpy as np
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from sklearn.metrics import silhouette_score

X = np.array([[0,0],[1,1],[10,10],[11,11]], dtype=float)
km = KMeans(k=2)
km.fit(X)
labels = km.predict(X)

sil = Silhouette()
scores = sil.score(X, labels)
# print(scores)
print(np.mean(scores))
sklearn_score = silhouette_score(X, labels)
print(sklearn_score)