import numpy as np
import pytest
from cluster.kmeans import KMeans
from cluster.utils import make_clusters

def test_assert_Errors():
    """
    Make sure all errors are raised properly in the kmeans.py
    """
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

def test_k_zero_raises():
    """
    test if i correctly raise an error when the user inputs k=0
    """
    with pytest.raises(ValueError, match="k must be a positive integer"):
        KMeans(k=0)


def test_kmeans():
    """
    does kmeans prediction work on very small dataset?
    """
    X = np.array([[0.0, 0.0],
                  [1.0, 1.0],
                  [10.0, 10.0],
                  [11.0, 11.0]])

    km = KMeans(k=2)
    km.fit(X)
    labels = km.predict(X)

    # Returns 1 label per data pt
    assert labels.shape == (X.shape[0],)
    # check that the right number of centroids is saved (k = 2 centroids)
    assert km.get_centroids().shape == (2, X.shape[1])
    # check error is a single numberic value
    assert isinstance(km.get_error(), float)

def test_kmeans_on_generated_data():
    """
    does algorithm work on larger dataset generated with the provided code?
    """
    points, true_labels = make_clusters(n=100, k=5)
    km = KMeans(k=5)
    km.fit(points)
    pred_labels = km.predict(points)
    centroids = km.get_centroids()
    err = km.get_error()
    assert len(centroids) == 5 # because i specified there are 5 clusters
    assert len(set(pred_labels)) == 5


