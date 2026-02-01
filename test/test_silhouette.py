import numpy as np
from cluster.kmeans import KMeans
from cluster.silhouette import Silhouette
import sklearn

def test_silhouette():
    X = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [10.0, 10.0],
                  [11.0, 11.0]])

    km = KMeans(k=2)
    km.fit(X)
    labels = km.predict(X)

    sil = Silhouette()
    scores = sil.score(X, labels)
    mean_my_score = np.mean(scores)
    # compare the score i get from my Silhouette class to sklearn.metrics.silhouette_score
    sklearn_score = sklearn.metrics.silhouette_score(X, labels)

    assert np.allclose(mean_my_score, sklearn_score, atol=1e-2)