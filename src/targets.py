import torch
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats

class TargetDistribution(ABC):
    """
    Abstract class for probability densities we can sample from.
    """
    
    @abstractmethod
    def sample(self, n : int) -> np.ndarray:
        """
        Gives n sample of density. 

        Args:
            n (int): amount of datapoints to sample

        Returns:
            np.ndarray: output samples [batch_size, dim]
        """
        pass
    

class LogDensity(ABC):
    """
    Abstract class for probability densities where we know log density and its score. 
    """
    @property
    @abstractmethod
    def log_prob(self, x : np.ndarray) -> np.ndarray:
        """
        Computes log(p(x))
        Args:
            x (torch.Tensor): some point of shape [batch_size, dim]

        Returns:
            torch.Tensor: log(p(x)) [batch_size, 1]
        """
        
        pass

    def score(self, x : np.ndarray) -> np.ndarray:
        """
        Returns the Gradient log density(x) i.e. the score
        Args:
            x: [batch_size, dim]
        Returns:
            score: [batch_size, dim]
        """
        x = torch.from_numpy(x).unsqueeze(1)  # [batch_size, 1, dim]
        score = torch.vmap(torch.func.jacrev(self.log_prob))(x)  # [batch_size, 1, dim, dim]
        return score.squeeze((1, 2, 3)).numpy()  # [batch_size, dim]


class Gaussian(LogDensity, TargetDistribution):
    """
    2D Gaussian distribution with numpy implementation.
    """
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        mean: shape [2,]
        cov: shape [2, 2]
        """
        super().__init__()
        self.mean = mean.astype(np.float64)
        self.cov = cov.astype(np.float64)
        
        # Precompute for efficiency
        self._cov_inv = np.linalg.inv(self.cov)
        self._cov_chol = np.linalg.cholesky(self.cov)
        
        # Log normalization constant: -0.5 * (2 * log(2*pi) + log(det(cov)))
        sign, logdet = np.linalg.slogdet(self.cov)
        self._log_norm = -0.5 * (2 * np.log(2 * np.pi) + logdet)

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Sample from 2D Gaussian using Cholesky decomposition.
        Returns: shape [num_samples, 2]
        """
        # Generate standard normal samples
        z = np.random.randn(num_samples, 2)
        # Transform: x = mean + L @ z, where L is Cholesky factor
        samples = self.mean + z @ self._cov_chol.T
        return samples
        
    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log probability density.
        x: shape [batch_size, 2]
        Returns: shape [batch_size, 1]
        """
        # Center the data
        diff = x - self.mean  # [batch_size, 2]
        
        # Mahalanobis distance: (x - mean)^T @ cov_inv @ (x - mean)
        mahalanobis = np.sum(diff @ self._cov_inv * diff, axis=1)  # [batch_size,]
        
        # Log probability: log_norm - 0.5 * mahalanobis
        log_prob = self._log_norm - 0.5 * mahalanobis
        
        return log_prob.reshape(-1, 1)

    @classmethod
    def isotropic(cls, std: float) -> "Gaussian":
        """Create isotropic 2D Gaussian with given std."""
        mean = np.zeros(2)
        cov = np.eye(2) * std ** 2
        return Gaussian(mean, cov)


class GaussianMixture(LogDensity, TargetDistribution):
    """
    2D Gaussian mixture model with numpy implementation.
    """
    def __init__(
        self,
        means: np.ndarray,  # nmodes x 2
        covs: np.ndarray,   # nmodes x 2 x 2
        weights: np.ndarray,  # nmodes
    ):
        """
        means: shape (nmodes, 2)
        covs: shape (nmodes, 2, 2)
        weights: shape (nmodes,)
        """
        super().__init__()
        self.means = means.astype(np.float64)
        self.covs = covs.astype(np.float64)
        
        # Normalize weights
        self.weights = weights.astype(np.float64)
        self.weights = self.weights / np.sum(self.weights)
        
        self.nmodes = means.shape[0]
        
        # Precompute for each mode
        self._cov_invs = np.array([np.linalg.inv(cov) for cov in self.covs])
        self._cov_chols = np.array([np.linalg.cholesky(cov) for cov in self.covs])
        
        # Precompute log normalization constants for each mode
        self._log_norms = np.zeros(self.nmodes)
        for k in range(self.nmodes):
            sign, logdet = np.linalg.slogdet(self.covs[k])
            self._log_norms[k] = -0.5 * (2 * np.log(2 * np.pi) + logdet)

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute log probability density of mixture.
        x: shape [batch_size, 2]
        Returns: shape [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        # Compute log probability for each component
        log_probs = np.zeros((batch_size, self.nmodes))
        
        for k in range(self.nmodes):
            # Center the data
            diff = x - self.means[k]  # [batch_size, 2]
            
            # Mahalanobis distance
            mahalanobis = np.sum(diff @ self._cov_invs[k] * diff, axis=1)
            
            # Log probability for component k
            log_probs[:, k] = self._log_norms[k] - 0.5 * mahalanobis + np.log(self.weights[k])
        
        # Log-sum-exp trick for numerical stability
        max_log_prob = np.max(log_probs, axis=1, keepdims=True)
        log_prob = max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob), axis=1, keepdims=True))
        
        return log_prob

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Sample from 2D Gaussian mixture.
        Returns: shape [num_samples, 2]
        """
        # Sample component indices according to mixture weights
        component_indices = np.random.choice(self.nmodes, size=num_samples, p=self.weights)
        
        # Sample from each component
        samples = np.zeros((num_samples, 2))
        for k in range(self.nmodes):
            mask = component_indices == k
            n_k = np.sum(mask)
            if n_k > 0:
                # Generate standard normal samples
                z = np.random.randn(n_k, 2)
                # Transform using Cholesky factor
                samples[mask] = self.means[k] + z @ self._cov_chols[k].T
        
        return samples

    @classmethod
    def random_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0, seed: int = 0
    ) -> "GaussianMixture":
        """Create random 2D Gaussian mixture."""
        np.random.seed(seed)
        means = (np.random.rand(nmodes, 2) - 0.5) * scale + x_offset * np.array([1.0, 0.0])
        covs = np.tile(np.eye(2), (nmodes, 1, 1)) * std ** 2
        weights = np.ones(nmodes)
        return GaussianMixture(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        """Create symmetric 2D Gaussian mixture arranged in a circle."""
        angles = np.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = np.stack([np.cos(angles), np.sin(angles)], axis=1) * scale + np.array([1.0, 0.0]) * x_offset
        covs = np.tile(np.eye(2), (nmodes, 1, 1)) * std ** 2
        weights = np.ones(nmodes) / nmodes
        return GaussianMixture(means, covs, weights)





