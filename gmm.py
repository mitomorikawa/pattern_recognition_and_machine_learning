import numpy as np
import matplotlib.pyplot as plt
from clustering import KMeansClustering
from distributions import Gaussian, Dirichlet, Wishart
from scipy.special import digamma, logsumexp


class GaussianMixtureModel:
    """ 
    This class implements a Gaussian Mixture Model (GMM) using the Expectation-Maximization (EM) algorithm or Variational Inference (VI).
    It supports
    • Fitting the model to data
    • Predicting the cluster for new data points
    """

    def __init__(self, k=2, max_iter=1000):
        """
        Parameters:
            k (int): Number of clusters.
            max_iter (int): Maximum number of iterations for the fitting process.
            method (str): Method to use for fitting ('EM' or 'VI').

        attributes:
            k, max_iter, method
            means: List[np.ndarray(D,)]
            precisions: List[np.ndarray(D, D)]
            weights: np.array(k,)
            responsibility: np.ndarray(N, k)
        """

        self.k = k
        self.max_iter = max_iter
        self.log_likelihoods = []
        self.lower_bounds = []
        

    def initialize_params(self, X):
        N, D = X.shape
        kmeans = KMeansClustering(k=self.k)
        kmeans.fit(X)
        self.responsibility = None
        self.means = kmeans.get_centroids()
        self.covariances = kmeans.get_covariances()
        self.precisions = [np.linalg.inv(cov) for cov in self.covariances]
        self.weights = np.ones(self.k) / self.k
        self.m_0 = np.array(kmeans.get_centroids())
        self.m_k = self.m_0.copy()
        self.beta_0 = np.array([1.0 for _ in range(self.k)])
        self.beta_k = self.beta_0.copy()
        self.nu_0 = np.array([D for _ in range(self.k)])
        self.nu_k = self.nu_0.copy()
        self.W_0_inv = np.array(kmeans.get_covariances())
        self.W_k_inv = self.W_0_inv.copy()
        self.alpha_0 = np.array([1.0 for _ in range(self.k)])
        self.alpha_k = self.alpha_0.copy()

    def fitEM(self, X):
        """ 
        Fit the GMM to the data X using the Expectation-Maximization (EM) algorithm.
        Parameters:
            X (np.ndarray): Input data of shape (N, D) where N is the number of samples and D is the number of features.

        Returns:
            None
        """

        N, D = X.shape
        self.initialize_params(X)
        for iteration in range(self.max_iter):
            gaussians = np.array([[Gaussian(self.means[j], self.covariances[j]).pdf(X[n])  for j in range(self.k)] for n in range(N)])
            # E step: Compute responsibilities
            self.responsibility = (self.weights.reshape(1,-1) * gaussians) / (np.sum(self.weights * gaussians, axis=1).reshape(-1, 1) + 1e-10)
            # M step: Update parameters
            new_means = []
            new_covariances = []
            new_weights = []
            for k in range(self.k):
                N_k = np.sum(self.responsibility[:, k])
                mean_k = np.sum(self.responsibility[:, k].reshape(-1, 1) * X, axis=0) / N_k
                cov_k = (self.responsibility[:, k].reshape(-1, 1) * (X - mean_k)).T @ (X - mean_k) / N_k
                weight_k = N_k / N
                new_means.append(mean_k)
                new_covariances.append(cov_k)
                new_weights.append(weight_k)
            self.means = new_means
            self.covariances = new_covariances
            self.weights = np.array(new_weights)
            log_likelihood = np.sum(np.log(np.sum(self.weights * gaussians, axis=1) + 1e-10))
            self.log_likelihoods.append(log_likelihood)
            if iteration > 0 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < 1e-6:
                break
    
    def visualize_log_likelihood(self):
        plt.plot(self.log_likelihoods)
        plt.xlabel('Iteration')
        plt.ylabel('Log Likelihood')
        plt.title('Log Likelihood over Iterations')
        plt.show()

    def predictEM(self, X):
        """ 
        Predict the cluster for each data point in X.
        Parameters:
            X (np.ndarray): Input data of shape (N, D) where N is the number of samples and D is the number of features.
        Returns:
            responsibilities (np.ndarray): Responsibilities for each data point and cluster of shape (N, k).
        """
        N, D = X.shape
        gaussians = np.array([[Gaussian(self.means[j], self.covariances[j]).pdf(X[n])  for j in range(self.k)] for n in range(N)])
        responsibilities = (self.weights.reshape(1,-1) * gaussians) / (np.sum(self.weights * gaussians, axis=1).reshape(-1, 1) + 1e-10)
        return responsibilities
    
    def get_params(self):
        """ 
        Get the parameters of the fitted GMM.
        Returns:
            means (List[np.ndarray(D,)]): List of mean vectors for each cluster.
            covariances (List[np.ndarray(D, D)]): List of covariance matrices for each cluster.
            weights (np.ndarray(k,)): Mixture weights for each cluster.
        """
        return self.means, self.covariances, self.weights

    def fitVI(self, X):
        """ 
        Fit the GMM to the data X using Variational Inference (VI).
        Parameters:
            X (np.ndarray): Input data of shape (N, D) where N is the number of samples and D is the number of features.

        Returns:
            None
        """
        N, D = X.shape
        self.initialize_params(X)
        
        for iteration in range(self.max_iter):
            # E step: Compute responsibilities
            self.W_0_inv = (self.W_0_inv + self.W_0_inv.transpose(0,2,1)) / 2 # ensure symmetry
            self.W_k_inv = (self.W_k_inv + self.W_k_inv.transpose(0,2,1)) / 2 # ensure symmetry
            self.ln_pi_tilda = digamma(self.alpha_k) - digamma(np.sum(self.alpha_k)) #(k,)
            self.ln_lambda_tilda = np.array([np.sum(digamma((self.nu_k[j] + 1 - np.arange(1, D + 1)) / 2)) + D * np.log(2) - np.log(np.linalg.det(self.W_k_inv[j])) for j in range(self.k)]) #(k,)
            diff = X[:, None, :] - self.m_k[None, :, :]
            mahalanobis = np.einsum('nkd, kde, nke -> nk', diff, np.linalg.inv(self.W_k_inv), diff)
            ex_mahalanobis = D / self.beta_k + self.nu_k * mahalanobis #(N, k)
            log_pho = self.ln_pi_tilda + 0.5 * self.ln_lambda_tilda - 0.5 * D * np.log(2*np.pi) - 0.5 * ex_mahalanobis
            log_pho_norm = log_pho - logsumexp(log_pho, axis=1, keepdims=True)
            self.responsibility = np.exp(log_pho_norm)

            # M step: Update parameters
            N_k = np.sum(self.responsibility, axis=0) #(k,)
            x_k = np.einsum('nk, nd -> kd', self.responsibility, X) / N_k.reshape(-1, 1) #(k, D)
            S_k = np.array([np.einsum('n, nd, ne -> de', self.responsibility[:, j], X - x_k[j], X - x_k[j]) / N_k[j] for j in range(self.k)]) #(k, D, D)
            self.alpha_k = self.alpha_0 + N_k
            self.beta_k = self.beta_0 + N_k
            self.m_k = (self.beta_0.reshape(-1, 1) * self.m_0 + N_k.reshape(-1, 1) * x_k) / self.beta_k.reshape(-1, 1)
            self.nu_k = self.nu_0 + N_k
            self.W_k_inv = np.array([self.W_0_inv[j] + N_k[j] * S_k[j] + (self.beta_0[j] * N_k[j]) / (self.beta_0[j] + N_k[j]) * np.outer(x_k[j] - self.m_0[j], x_k[j] - self.m_0[j]) for j in range(self.k)])

            #compute lower bound
            ex_ln_X_given_Z_means_precisions = 0.5 * np.sum(N_k * (self.ln_lambda_tilda - D / self.beta_k - self.nu_k * np.array([np.trace(S_k[j] @ np.linalg.inv(self.W_k_inv[j])) for j in range(self.k)]) \
                                - self.nu_k * np.einsum('kd, kde, ke -> k', x_k - self.m_k, np.linalg.inv(self.W_k_inv), x_k - self.m_k) - D * np.log(2 * np.pi)))
            ex_ln_Z_given_pi = np.sum(self.responsibility * self.ln_pi_tilda)
            ex_ln_pi = Dirichlet(self.weights, self.alpha_0).logC() + (self.alpha_0[0] - 1) * np.sum(self.ln_pi_tilda)
            ex_ln_means_precisions = 0.5 * np.sum(D*np.log(self.beta_0 / (2 * np.pi)) + self.ln_lambda_tilda - D * self.beta_0 / self.beta_k - self.beta_0 * self.nu_k * np.einsum('kd, kde, ke -> k', \
                self.m_k - self.m_0, np.linalg.inv(self.W_k_inv), self.m_k - self.m_0)) + self.k * Wishart(self.W_0_inv[0], self.nu_0[0]).logB() + 0.5 * (self.nu_0[0] - D - 1) * np.sum(self.ln_lambda_tilda) - 0.5 * np.sum(self.nu_k * np.array([np.trace(self.W_0_inv[j] @ np.linalg.inv(self.W_k_inv[j])) for j in range(self.k)]))
            ex_ln_Z_approx = np.sum(self.responsibility * np.log(self.responsibility + 1e-10))
            ex_ln_pi_approx = np.sum((self.alpha_k - 1) * self.ln_pi_tilda) + Dirichlet(self.weights, self.alpha_k).logC()
            ex_ln_means_precisions_approx = np.sum(0.5 * self.ln_lambda_tilda + D * 0.5 * np.log(self.beta_k / (2 * np.pi)) - D / 2 + Wishart(self.W_k_inv, self.nu_k).logB() + 0.5 * (self.nu_k - D - 1) * self.ln_lambda_tilda - 0.5 * self.nu_k * D)
            lower_bound = ex_ln_X_given_Z_means_precisions + ex_ln_pi + ex_ln_Z_given_pi + ex_ln_means_precisions - ex_ln_Z_approx - ex_ln_pi_approx - ex_ln_means_precisions_approx
            self.lower_bounds.append(lower_bound)
            if iteration > 0 and abs(self.lower_bounds[-1] - self.lower_bounds[-2]) < 1e-6:
                break

    def visualize_lower_bound(self):
        plt.plot(self.lower_bounds)
        plt.xlabel('Iteration')
        plt.ylabel('Lower Bound')
        plt.title('Lower Bound over Iterations')
        plt.show()

    def predictVI(self, X):
        """ 
        Predict the cluster for each data point in X.
        Parameters:
            X (np.ndarray): Input data of shape (N, D) where N is the number of samples and D is the number of features.
        Returns:
            responsibilities (np.ndarray): Responsibilities for each data point and cluster of shape (N, k).
        """
        N, D = X.shape
        diff = X[:, None, :] - self.m_k[None, :, :]
        mahalanobis = np.einsum('nkd, kde, nke -> nk', diff, np.linalg.inv(self.W_k_inv), diff)
        ex_mahalanobis = D / self.beta_k + self.nu_k * mahalanobis #(N, k)
        log_pho = self.ln_pi_tilda + 0.5 * self.ln_lambda_tilda - 0.5 * D * np.log(2*np.pi) - 0.5 * ex_mahalanobis
        log_pho_norm = log_pho - logsumexp(log_pho, axis=1, keepdims=True)
        responsibilities = np.exp(log_pho_norm)
        return responsibilities