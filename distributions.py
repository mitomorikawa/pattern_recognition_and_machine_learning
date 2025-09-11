import numpy as np
from scipy.special import gamma, gammaln

class Gaussian:
    def __init__(self, mean, covariance):
        """ 
        attributes:
            mean (np.ndarray(1, D)): mean vector
            covariance (np.ndarray(D, D)): covariance matrix
            precision (np.ndarray(D, D)): precision matrix
            D: int

        """
        self.mean = mean
        self.covariance = covariance
        self.precision = np.linalg.inv(covariance)

    def logpdf(self, x):
        """ 
        Compute the log of the probability density function (PDF) of the Gaussian at point x.
        Parameters:
            x (np.ndarray(1, D)): Input data point.
        Returns:
            logpdf_value (float): The computed log PDF value.
        """
        D = self.mean.shape[0]
        diff = x - self.mean
        sign, logdet = np.linalg.slogdet(self.covariance)
        return -0.5 * (diff @ self.precision @ diff.T).item() - 0.5 * (D * np.log(2 * np.pi) + logdet)
    
    def pdf(self, x):
        """ 
        Compute the probability density function (PDF) of the Gaussian at point x.
        Parameters:
            x (np.ndarray(1, D)): Input data point.
        Returns:
            pdf_value (float): The computed PDF value.
        """
        return np.exp(self.logpdf(x))
        
class Dirichlet:
    def __init__(self, mu_k, alpha_k):
        """ 
        attributes:
            mu_k (np.ndarray(k,)): mean vector
            alpha_k (np.ndarray(k,)): concentration parameters
            k: int

        """
        self.mu_k = mu_k
        self.alpha_k = alpha_k

    def logC(self):
        """ 
        Compute the log normalization constant lnC of the Dirichlet distribution.
        Returns:
            lnC (float): The computed log normalization constant.
        """
        return gammaln(np.sum(self.alpha_k)) - np.sum(gammaln(self.alpha_k))
    
    def logpdf(self):
        """ 
        Compute the log of the probability density function (PDF) of the Dirichlet distribution.
        Returns:
            logpdf_value (float): The computed log PDF value.
        """
        return self.logC() + np.sum((self.alpha_k - 1) * np.log(self.mu_k))
    
    def pdf(self):
        """ 
        Compute the probability density function (PDF) of the Dirichlet distribution.
        Returns:
            pdf_value (float): The computed PDF value.
        """
        return np.exp(self.logpdf())
    
class Wishart:
    def __init__(self, W_inv, nu):
        """ 
        attributes:
            W_inv (np.ndarray(D, D)): scale matrix (inverse of the covariance matrix)
            nu (int): degrees of freedom
            D: int

        """
        self.W_inv = W_inv
        self.nu = nu
    
    def logB(self):
        """ 
        Compute the normalization constant B of the Wishart distribution.
        Returns:
            B (float): The computed normalization constant.
        """
        D = self.W_inv.shape[0]
        sign, logdet = np.linalg.slogdet(self.W_inv)
        return (self.nu / 2) * logdet -(self.nu * D / 2) * np.log(2)  - (D * (D - 1) / 4) * np.log(np.pi) \
               - np.sum([gammaln((self.nu + 1 - i) / 2) for i in range(1, D + 1)])
    
    def logpdf(self, X):
        """ 
        Compute the log of the probability density function (PDF) of the Wishart distribution at matrix X.
        Parameters:
            X (np.ndarray(D, D)): Input data matrix.
        Returns:
            logpdf_value (float): The computed log PDF value.
        """
        D = self.W_inv.shape[0]
        sign, logdetX = np.linalg.slogdet(X)
        return self.logB() + ((self.nu - D - 1) / 2) * logdetX - 0.5 * np.trace(self.W_inv @ X)
    
    def pdf(self, X):
        """ 
        Compute the probability density function (PDF) of the Wishart distribution at matrix X.
        Parameters:
            X (np.ndarray(D, D)): Input data matrix.
        Returns:
            pdf_value (float): The computed PDF value.
        """
        return np.exp(self.logpdf(X))