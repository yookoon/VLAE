import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import distribution


class BaseEncoder(nn.Module):
    def __init__(self, z_dim, x_dim, h_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.linear_hidden0 = nn.Linear(self.x_dim[1] * self.x_dim[2] * self.x_dim[0], h_dim)
        self.linear_hidden1 = nn.Linear(h_dim, h_dim)
        self.linear_mu = nn.Linear(h_dim, z_dim)
        self.linear_logvar = nn.Linear(h_dim, z_dim)
        self.activation = F.relu

    def forward(self, x):
        h = self.activation(self.linear_hidden0(x))
        h = self.activation(self.linear_hidden1(h))
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)

        return distribution.DiagonalGaussian(mu, logvar), h

class BaseDecoder(nn.Module):
    def __init__(self, z_dim, output_dist, x_dim, h_dim):
        super().__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.linear_hidden0 = nn.Linear(z_dim, h_dim)
        self.linear_hidden1 = nn.Linear(h_dim, h_dim)
        self.linear_mu = nn.Linear(h_dim, self.x_dim[1] * self.x_dim[2] * self.x_dim[0])
        self.activation = F.relu
        self.output_dist = output_dist

        if output_dist == 'gaussian':
            self.logvar = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, z, compute_jacobian=False):
        if compute_jacobian:
            h = self.activation(self.linear_hidden0(z))

            # activation_mask: [batch_size, hidden_dim, 1]
            activation_mask = torch.unsqueeze(h > 0.0, dim=-1).to(torch.float)
            # W: [hidden_dim, input_dim]
            W = self.linear_hidden0.weight
            # W: [batch_size, hidden_dim, input_dim]
            W = activation_mask * W

            h = self.activation(self.linear_hidden1(h))
            activation_mask = torch.unsqueeze(h > 0.0, dim=-1).to(torch.float)
            W = torch.matmul(self.linear_hidden1.weight, W)
            W = activation_mask * W

            W = torch.matmul(self.linear_mu.weight, W)

            mu = self.linear_mu(h)
            W_out = W

            if self.output_dist == 'gaussian':
                return distribution.DiagonalGaussian(mu, self.logvar), W_out
            elif self.output_dist == 'bernoulli':
                mu = torch.sigmoid(mu)
                mu_clip = torch.clamp(mu, min=1e-5, max=1.0 - 1e-5)
                self.logvar = -torch.log((mu_clip * (1 - mu_clip)) + 1e-5)
                return distribution.Bernoulli(mu), W_out
            else:
                raise ValueError

        else:
            h = self.activation(self.linear_hidden0(z))
            h = self.activation(self.linear_hidden1(h))
            mu = self.linear_mu(h)

            if self.output_dist == 'gaussian':
                return distribution.DiagonalGaussian(mu, self.logvar)
            elif self.output_dist == 'bernoulli':
                mu = torch.sigmoid(mu)
                return distribution.Bernoulli(mu)
            else:
                raise ValueError


class HouseHolderFlow(nn.Module):
    def __init__(self, h_dim, z_dim, n_flow):
        super().__init__()
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.linear_flows = nn.ModuleList([nn.Linear(h_dim, z_dim) if i == 0
                                           else nn.Linear(z_dim, z_dim)
                                           for i in range(n_flow)])
        self.I = torch.eye(self.z_dim).unsqueeze(0).cuda()

    def forward(self, h):
        H = self.I.clone()
        for linear_flow in self.linear_flows:
            h = linear_flow(h)
            H = torch.matmul(self.I - 2.0 * torch.matmul(h.unsqueeze(-1),
                                                         h.unsqueeze(1))
                             / h.pow(2).sum(dim=-1, keepdim=True).unsqueeze(-1),
                             H)

        return H


class MADE(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        2-Layer MADE (http://arxiv.org/abs/1502.03509)
        Used as AutoregressiveNN for Inverse Autogressive Flow (IAF)
        """

        super(MADE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W = nn.Linear(self.input_size + self.hidden_size, self.hidden_size, bias=False)
        self.b = nn.Parameter(torch.randn(self.hidden_size))

        self.V_s = nn.Linear(self.hidden_size, self.input_size, bias=False)
        self.V_m = nn.Linear(self.hidden_size, self.input_size, bias=False)
        self.c_s = nn.Parameter(torch.ones(self.input_size) * 2.0)
        self.c_m = nn.Parameter(torch.randn(self.input_size))

        self.W_mask, self.V_mask = self.generate_mask()

        self.relu = nn.PReLU()

    def generate_mask(self):
        """Generate masks for network weights"""
        # m(k)
        # randomly generate the indexes.
        # Q: shouldn't this be input_size - 1? A: it is. arg high discounts itself.
        max_masks = np.random.randint(low=1, high=self.input_size, size=self.hidden_size)

        # M^W
        # note: input_size + hidden_size b/c z and h are concatted as the input.
        W_mask = np.fromfunction(
            lambda k, d: max_masks[k] >= d + 1, (self.hidden_size, self.input_size + self.hidden_size),
            dtype=int).astype(np.float32)
        W_mask = nn.Parameter(torch.from_numpy(W_mask), requires_grad=False)

        # M^V
        V_mask = np.fromfunction(
            lambda d, k: d + 1 > max_masks[k], (self.input_size, self.hidden_size),
            dtype=int).astype(np.float32)
        V_mask = nn.Parameter(torch.from_numpy(V_mask), requires_grad=False)

        # Check strict lower triangular
        # M^V @ M^W must be strictly lower triangular
        # => M^V @ M^W 's upper triangular is zero matrix
        assert ((V_mask.data @ W_mask.data).triu().eq(
            torch.zeros(self.input_size, self.input_size + self.hidden_size))).all()

        return W_mask, V_mask

    def apply_mask(self):
        """Mask weights"""
        self.W.weight.data = (self.W.weight * self.W_mask).data
        self.V_s.weight.data = (self.V_s.weight * self.V_mask).data
        self.V_m.weight.data = (self.V_m.weight * self.V_mask).data

    def forward(self, z, h):
        """
        Args:
            z: [batch_size, z_size]
            h: [batch_size, h_size]
            input_size = z_size + h_size
        Return
            m: [batch_size, input_size]
            s: [batch_size, input_size]
        """

        self.apply_mask()
        x = self.W(torch.cat([z, h], dim=1)) + self.b
        x = self.relu(x)

        m = self.V_m(x) + self.c_m
        s = self.V_s(x) + self.c_s

        return m, s
