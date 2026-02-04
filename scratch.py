import numpy as np
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
from sklearn.metrics import silhouette_score


from cluster.utils import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)



# X = np.array([[0,0],[1,1],[10,10],[11,11]], dtype=float)
# km = KMeans(k=2)
# km.fit(X)
# labels = km.predict(X)


X, labels_true = make_clusters(n=10000, k=3) # outputs mat, labels

km = KMeans(k=3)
km.fit(X)
labels_pred = km.predict(X)
centroids = km.get_centroids()
err = km.get_error()

print(len(set(labels_pred)))
print(np.shape(labels_pred))
print(set(labels_pred))

sil = Silhouette()
scores = sil.score(X, labels_pred)
print(np.mean(scores))
sklearn_score = silhouette_score(X, labels_pred)
print(sklearn_score)

# pred_counts = np.bincount(labels_pred, minlength=3)
# print("predicted label counts:", pred_counts)
# true_counts = np.bincount(labels_true, minlength=3)
# print("true label counts:", true_counts)


 
# print(labels_pred.shape == (X.shape[0],))
# print(km.get_centroids().shape == (3, X.shape[1]))