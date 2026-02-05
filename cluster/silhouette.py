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
        n = X.shape[0] # n is the observations ie the rows 
        labels = np.unique(y) 
        # if k = 1 this scoring func will not work
        if labels.size < 2:
            raise ValueError("Number of clusters is 1. Valid values begin at 2.")


        # calc all pairwise distances. store in D so i can index these later instead of needing to compute them in the for loop
        D = cdist(X, X, metric='euclidean') # D[i, j] = distance between point i and point j

        #initialize output
        silhouette = np.zeros(n) 

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
            b = np.inf # if k = 1 this will be a problem
            for other_label in labels: # loop over all other clusters
                if other_label == label_:
                    continue

                other_cluster_idx = np.where(y == other_label)[0] # points belonging to other clusters
                mean_dist = D[i, other_cluster_idx].mean()
                b = min(b, mean_dist)


            # silhouette score
            denom = max(a,b)
            if denom == 0.0:
                silhouette[i] = 0.0 # to avoid dividing by 0
            else:
                silhouette[i] = (b - a)/denom

        return silhouette
