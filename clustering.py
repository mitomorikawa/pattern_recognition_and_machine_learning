import numpy as np
import matplotlib.pyplot as plt


class KMeansClustering:
    """ 
    This class implements the K-Means clustering algorithm.

    It supports:
    • Fitting the model to data
    • Tracking the cost function over iterations
    • Predicting the cluster for new data points
    """

    def __init__(self, k=2, max_iter=1000):
        """
        Parameters:
            k (int): Number of clusters.
            max_iter (int): Maximum number of iterations for the fitting process.

        Attributes:
            k (int): Number of clusters.
            max_iter (int): Maximum number of iterations.
            centroids (List[np.ndarray]): List of centroid positions.
            covariances (List[np.ndarray]): List of covariance matrices for each cluster.
            responsibility (np.ndarray): Responsibility matrix.(each element is 1 or 0)   
            log_likelihoods (list): Log-likelihood values over iterations.
        """

        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.covariances = []
        self.responsibility = None
        self.costs = []

    def compute_cost(self, X):
        """ 
        Compute the cost (sum of squared distances) of the current clustering.
        Parameters:
            X (np.ndarray): Input data of shape (N, D).
        Returns:
            cost (float): The computed cost.
        """
        N, D = X.shape
        cost = 0.0
        for i in range(N):
            for j in range(self.k):
                if self.responsibility[i, j] == 1:
                    cost += np.linalg.norm(X[i] - self.centroids[j]) ** 2
        return cost

    def fit(self, X):
        """ 
        Fit the K-Means model to the data X.
        Parameters:
            X (np.ndarray): Input data of shape (N, D) where N is the number of samples and D is the number of features.
        Returns:
            None
        """
        N, D = X.shape
        # Initialize centroids randomly from the data points
        random_indices = np.random.choice(N, self.k, replace=False)
        self.centroids = [X[idx] for idx in random_indices]

        for iteration in range(self.max_iter):
            # E-step: Assign clusters
            self.responsibility = np.zeros((N, self.k))
            for i in range(N):
                distances = [np.linalg.norm(X[i] - centroid) for centroid in self.centroids]
                closest_centroid = np.argmin(distances)
                self.responsibility[i, closest_centroid] = 1

            # M-step: Update centroids
            new_centroids = []
            for j in range(self.k):
                points_in_cluster = X[self.responsibility[:, j] == 1]
                if len(points_in_cluster) > 0:
                    new_centroid = np.mean(points_in_cluster, axis=0)
                else:
                    new_centroid = self.centroids[j]
                new_centroids.append(new_centroid)
        
            self.centroids = new_centroids
            # Compute cost
            cost = self.compute_cost(X)
            self.costs.append(cost)
            # Check for convergence (if cost does not change)
            if iteration > 0 and abs(self.costs[-1] - self.costs[-2]) < 1e-6:
                break
        
        # Compute covariances for each cluster for later use
        for j in range(self.k):
            points_in_cluster = X[self.responsibility[:, j] == 1]
            if len(points_in_cluster) > 1:
                covariance = np.cov(points_in_cluster, rowvar=False)
            else:
                covariance = np.eye(D)
            self.covariances.append(covariance)


    def visualise_costs(self):
        """ 
        Visualize the cost over iterations.
        """
        plt.plot(self.costs)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('K-Means Cost over Iterations')
        plt.show()
            
    def predict(self, X):
        """ 
        Predict the cluster for new data points X.
        Parameters:
            X (np.ndarray): Input data of shape (M, D) where M is the number of samples and D is the number of features.
        Returns:
            predictions (np.ndarray): Predicted cluster indices for each data point.
        """
        M, D = X.shape
        predictions = np.zeros(M, dtype=int)
        for i in range(M):
            distances = [np.linalg.norm(X[i] - centroid) for centroid in self.centroids]
            predictions[i] = np.argmin(distances)
        return predictions

    def get_centroids(self):
        """ 
        Get the current centroids of the clusters.
        Returns:
            centroids (List[np.ndarray]): List of centroid positions.
        """
        return self.centroids
    
    def get_covariances(self):
        """ 
        Get the covariance matrices of the clusters.
        Returns:
            covariances (List[np.ndarray]): List of covariance matrices for each cluster.
        """
        return self.covariances