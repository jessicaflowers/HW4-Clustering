import numpy as np
import pytest
from cluster.kmeans import KMeans
from cluster.utils import (
  make_clusters, 
  plot_clusters, 
  plot_multipanel)

def test_assert_Errors():
    # input not np array
    mat = [
        [1, 2, 3],
        [4, 5, 6]
    ]
    km = KMeans(k=2)
    with pytest.raises(TypeError, match="input matrix must be a numpy array"):
        km.fit(mat)

    # input not 2d
    mat = np.array([1, 2, 3, 4, 5])
    km = KMeans(k=2)
    with pytest.raises(ValueError, match="input matrix must be a 2D array"):
        km.fit(mat)

    # k is larger than the number of data points
    mat = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [10.0, 10.0],
                  [11.0, 11.0]])
    km = KMeans(k=5)
    with pytest.raises(ValueError, match="number of samples must be >= k"):
        km.fit(mat)


    # predict is called before fit
    mat = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [10.0, 10.0],
                  [11.0, 11.0]])
    km = KMeans(k=2)
    with pytest.raises(RuntimeError, match="must call fit\(\) before predicting"):
        km.predict(mat)

    # get_error is called before fit
    mat = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [10.0, 10.0],
                  [11.0, 11.0]])
    km = KMeans(k=2)
    with pytest.raises(RuntimeError, match="model has not been fit yet; no error"):
        km.get_error()

    # get_centroids is called before fit
    mat = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [10.0, 10.0],
                  [11.0, 11.0]])
    km = KMeans(k=2)
    with pytest.raises(RuntimeError, match="model has not been fit yet; no centroids"):
        km.get_centroids()

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

def test_kmeans_on_generated_data():
    points, true_labels = make_clusters(n=7000)
    km = KMeans(k=4)
    km.fit(points)
    pred_labels = km.predict(points)
    centroids = km.get_centroids()
    err = km.get_error()
    assert len(centroids) == 4 # because i specified there are 4 clusters
    assert len(set(pred_labels)) == 4 
    # len(set(k_clusts))
    # compare the original clusters to my kmeans clusters




