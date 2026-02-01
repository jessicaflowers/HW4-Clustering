import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        # for i in each datapoint:
            #what i want: a_i = intra-cluster distance --> calculate the distance between this datapoint and all other points in the same cluster
            # how to do it:
            # given datpoint i, extract the label (the cluster it belongs to)
            # given that label (cluster), extract all the other datapoints belonging to that cluster
            # for every point belonging to that cluster, calculate the distance
            # take the mean of all these distances
            # what i want: b_i = nearest-cluster distance --> the average distance between the data point and all points in the nearest neighbouring cluster
        # silhouette score = (bi - ai)/max(ai,bi)
        # return silhouette_score

        n = X.shape[0]
        labels = np.unique(y)

        # calc all pairwise distances
        D = cdist(X, X, metric='euclidean') # D[i, j] = distance between point i and point j

        #initialize output
        silhouette = np.zeros(n) # do i really want float??

        # for every data point, i need to calculate the intra-cluster distance (a_i) and the nearest-cluster distance (b_i), which i use to compute the score
        for i in range(n):
            label_ = y[i] # this is the label (ie the cluster) of the datapoint i
            # datapoints in same cluster
            same_cluster_idx = np.where(y==label_)[0]
            # intra-cluster
            if same_cluster_idx.size <=1:
                a = 0.0
            else:
                dist_a_temp = D[i, same_cluster_idx] # distances from i to all points in same cluster
                dist_a = dist_a_temp[same_cluster_idx != i] # exclude distances from i to itself
                a = dist_a.mean()

            # nearest-cluster distance
            b = np.inf
            for other_label in labels: # loop over all other clusters
                if other_label == label_:
                    continue

                other_cluster_idx = np.where(y == other_label)[0] # points belonging to other clusters
                dist_b = D[i, other_cluster_idx]
                b = dist_b.mean()

            # silhouette score
            denom = max(a,b)
            if denom == 0.0:
                silhouette[i] = 0.0
            else:
                silhouette[i] = (b - a)/denom

        return silhouette
