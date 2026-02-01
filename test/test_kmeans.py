import numpy as np
from cluster.kmeans import KMeans

def test_kmeans():
    X = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [10.0, 10.0],
                  [11.0, 11.0]])

    km = KMeans(k=2)
    km.fit(X)
    labels = km.predict(X)

    assert labels.shape == (X.shape[0],)
    assert km.get_centroids().shape == (2, X.shape[1])
    assert isinstance(km.get_error(), float)
