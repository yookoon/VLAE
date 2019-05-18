import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import utils


class Bernoulli():
    def __init__(self, mu):
        self.mu = mu

    def log_probability(self, x):
        self.mu = torch.clamp(self.mu, min=1e-5, max=1.0 - 1e-5)
        return (x * torch.log(self.mu) + (1.0 - x) * torch.log(1 - self.mu)).sum(1)

    def sample(self):
        return (torch.rand_like(self.mu).to(device=self.mu.device) < self.mu).to(torch.float)


class DiagonalGaussian():
    def __init__(self, mu, logvar):
        self.mu = mu
        self.logvar = logvar

    def log_probability(self, x):
        return -0.5 * torch.sum(np.log(2.0*np.pi) + self.logvar + ((x - self.mu)**2)
                                / torch.exp(self.logvar), dim=1)

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + torch.exp(0.5 * self.logvar) * eps

    def repeat(self, n):
        mu = self.mu.unsqueeze(1).repeat(1, n, 1).view(-1, self.mu.shape[-1])
        logvar = self.logvar.unsqueeze(1).repeat(1, n, 1).view(-1, self.logvar.shape[-1])
        return DiagonalGaussian(mu, logvar)

    @staticmethod
    def kl_div(p, q):
        return 0.5 * torch.sum(q.logvar - p.logvar - 1.0 + (torch.exp(p.logvar) + (p.mu - q.mu)**2)/(torch.exp(q.logvar)), dim=1)


class Gaussian():
    def __init__(self, mu, precision):
        # mu: [batch_size, z_dim]
        self.mu = mu
        # precision: [batch_size, z_dim, z_dim]
        self.precision = precision
        # TODO: get rid of the inverse for efficiency
        self.L = torch.cholesky(torch.inverse(precision))
        self.dim = self.mu.shape[1]

    def log_probability(self, x):
        indices = np.arange(self.L.shape[-1])
        return -0.5 * (self.dim * np.log(2.0*np.pi)
                       + 2.0 * torch.log(self.L[:, indices, indices]).sum(1)
                       + torch.matmul(torch.matmul((x - self.mu).unsqueeze(1), self.precision),
                                      (x - self.mu).unsqueeze(-1)).sum([1, 2]))

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + torch.matmul(self.L, eps.unsqueeze(-1)).squeeze(-1)

    def repeat(self, n):
        mu = self.mu.unsqueeze(1).repeat(1, n, 1).view(-1, self.mu.shape[-1])
        precision = self.precision.unsqueeze(1).repeat(1, n, 1, 1).view(-1, *self.precision.shape[1:])
        return Gaussian(mu, precision)
