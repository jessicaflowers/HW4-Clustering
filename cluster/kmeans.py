import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        # if user provides wrong type of input for k
        if not isinstance(k, int) or k<=0:
            raise ValueError('k must be a positive integer')

        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        self.centroids = None
        self.error = None


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        # randomly select k rows from mat to initialize the cluster centroids
        # loop up to max_iter times
        # for each xi datapoint
            # find nearest centroid cj --> euclidean distance
            # assign xi to cluster j
        # for each cluster j = 1...k
            # compute the new centroid center cj = mean of all points xi assigned to cluster j in previous step
            # compute the change between the old and new centroid
            # stop early if the biggest change is below tol
        # save the final centroids as self.centroid 
        # save the final error (the averaged square distance from each point to its centroid)

        if not isinstance(mat, np.ndarray):
            raise TypeError('input matrix must be a numpy array')

        if mat.ndim != 2:
            raise ValueError("input matrix must be a 2D array")
        
        
        n_samples, n_features = mat.shape
        if n_samples < self.k:
            raise ValueError("number of datapoints must be >= k")
        

        # ranomly place a centroid for each of the k clusters
        rng = np.random.default_rng() # random number generator 
        random_ind = rng.choice(n_samples, size=self.k, replace=False)
        centroids = mat[random_ind]

        for iter in range(self.max_iter):
            # find nearest cetroid to each datapoint
            dist = cdist(mat, centroids,  metric='euclidean')
            # assign each data point to the closest centroid
            labels = np.argmin(dist, axis=1)

            # re calculate the center of mass for the updated centroid
            new_com = np.zeros_like(centroids) # initialize new matrix
            for j in range(self.k):
                # the new centroid needs to be the mean of all points assigned to that cluster in prev step, so extract the assigned data points for this cluster j
                cluster_points = mat[labels==j] 
                new_com[j] = np.mean(cluster_points, axis=0)

            # check for convergence. end the loop either if iter reaches max iter, or the cluster assignments do not change anymore within some tolerance
            centroid_shift = np.linalg.norm(new_com - centroids, axis=1)
            max_shift = np.maxx(centroid_shift)
            # update centroid
            centroids = new_com
            if max_shift < self.tol:
                break

        self.centroids = centroids # save final coordinates of the k centroids

        final_dist = cdist(mat, self.centroids, metric='euclidean')
        min_dist = np.min(final_dist, axis=1)
        self.error = np.mean(min_dist ** 2)



    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if self.centroids is None:
            raise RuntimeError("must call fit() before predicting")
        

        distances = cdist(mat, self.centroids, metric='euclidean')
        labels = np.argmin(distances, axis=1)

        return labels


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        if self.error is None:
            raise RuntimeError("model has not been fit yet; no error")
        
        return self.error

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        if self.centroids is None:
            raise RuntimeError("model has not been fit yet; no centroids")
        return self.centroids
