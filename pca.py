import numpy as np

class PCA:
    """
    This class implements PCA. 
    It supports:
    - Fitting the PCA model to the data
    - Transforming data to the PCA space
    """
    def __init__(self, n_components):
        """
        attributes:
            n_components (int): number of principal components to keep
            components_ (np.ndarray(D, n_components)): principal components
            mean_ (np.ndarray(D,)): mean of the data
            cov_ (np.ndarray(D, D)): covariance matrix of the data
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.cov_ = None

    def fit(self, X):
        """
        Fit the PCA model to the data.
        Args:
            X (np.ndarray(N, D)): Input data matrix.
        Returns:
            self
        """
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Compute the covariance matrix
        self.cov_ = np.cov(X_centered, rowvar=False)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(self.cov_)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[sorted_indices]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select the top n_components
        self.components_ = sorted_eigenvectors[:, :self.n_components]
        print(self.components_.shape)
        return self

    def transform(self, X):
        """ 
        Transform the data to the PCA space.
        Args:
            X (np.ndarray(N, D)): Input data matrix.
        Returns:
            X_pca (np.ndarray(N, n_components)): Transformed data in PCA space.
        """

        X_pca = (X - self.mean_) @ self.components_
        return X_pca