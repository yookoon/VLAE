import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np
import distribution
import network
import utils

n_importance_sample = 5000

class VAE(nn.Module):
    def __init__(self, dataset, z_dim, output_dist, x_dim, enc_dim, dec_dim, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.z_dim = z_dim
        self.output_dist = output_dist
        self.encoder = network.BaseEncoder(z_dim=z_dim, x_dim=x_dim, h_dim=enc_dim)
        self.decoder = network.BaseDecoder(z_dim=z_dim, x_dim=x_dim, h_dim=dec_dim, output_dist=output_dist)
        self.prior = distribution.DiagonalGaussian(mu=torch.zeros(1,1).cuda(), logvar=torch.zeros(1,1).cuda())
        self.x_dim = x_dim
        self.image_size = x_dim[0] * x_dim[1] * x_dim[2]

    def forward(self, x):
        q_z_x, _ = self.encoder(x) # returns distribution
        z = q_z_x.sample()
        p_x_z = self.decoder(z)
        return self.loss(x, z, p_x_z, self.prior, q_z_x)

    def loss(self, x, z, p_x_z, p_z, q_z_x):
        return -torch.mean(p_x_z.log_probability(x)
                           + p_z.log_probability(z)
                           - q_z_x.log_probability(z))

    def importance_weighting(self, x, z, p_x_z, p_z, q_z_x):
        log_weights = (p_x_z.log_probability(x)
                       + p_z.log_probability(z)
                       - q_z_x.log_probability(z)).view(-1, n_importance_sample)
        m = log_weights.max(1, keepdim=True)[0]
        weights = torch.exp(log_weights - m)
        loglikelihood = torch.mean(torch.log(weights.mean(dim=1)) + m, 1).sum()
        return loglikelihood

    def importance_sample(self, x):
        q_z_x, _ = self.encoder(x) # returns distribution
        q_z_x = q_z_x.repeat(n_importance_sample)
        z = q_z_x.sample()
        p_x_z = self.decoder(z)
        x = x.unsqueeze(1).repeat(1, n_importance_sample, 1).view(-1, x.shape[-1])
        return self.importance_weighting(x, z, p_x_z, self.prior, q_z_x)

    def write_summary(self, x, writer, epoch):
        with torch.no_grad():
            q_z_x, _ = self.encoder.forward(x)
            z = q_z_x.sample()
            p_x_z = self.decoder.forward(z)

            writer.add_scalar('kl_div',
                              torch.mean(-self.prior.log_probability(z)
                                         + q_z_x.log_probability(z)).item(),
                              epoch)
            writer.add_scalar('recon_error',
                              -torch.mean(p_x_z.log_probability(x)).item(),
                              epoch)
            writer.add_image('data',
                             vutils.make_grid(self.dataset.unpreprocess(x)),
                             epoch)
            writer.add_image('reconstruction_z',
                             vutils.make_grid(self.dataset.unpreprocess(p_x_z.mu).clamp(0, 1)),
                             epoch)

            sample = torch.randn(len(x), z.shape[1]).cuda()
            sample = self.decoder(sample).mu
            writer.add_image('generated',
                             vutils.make_grid(self.dataset.unpreprocess(sample).clamp(0, 1)),
                             epoch)


class VLAE(VAE):
    def __init__(self, dataset, z_dim, output_dist, x_dim, enc_dim, dec_dim,
                 n_update, update_lr, **kwargs):
        super().__init__(dataset, z_dim, output_dist, x_dim, enc_dim, dec_dim)
        self.n_update=n_update
        self.update_lr=update_lr

    def update_rate(self, t):
        return self.update_lr / (t+1)

    def solve_mu(self, x, mu_prev, p_x_z, W_dec):
        var_inv = torch.exp(-self.decoder.logvar).unsqueeze(1)
        precision = torch.matmul(W_dec.transpose(1, 2) * var_inv, W_dec)
        precision += torch.eye(precision.shape[1]).unsqueeze(0).cuda()

        if self.output_dist == 'gaussian':
            bias = p_x_z.mu.unsqueeze(-1) - torch.matmul(W_dec, mu_prev.unsqueeze(-1))
            mu = torch.matmul(W_dec.transpose(1, 2) * var_inv, x.view(-1, self.image_size, 1) - bias)
            mu = torch.matmul(torch.inverse(precision), mu)
            mu = mu.squeeze(-1)
        elif self.output_dist == 'bernoulli':
            bias = p_x_z.mu.unsqueeze(-1) - torch.matmul(W_dec * var_inv.transpose(1, 2), mu_prev.unsqueeze(-1))
            mu = torch.matmul(W_dec.transpose(1, 2), x.view(-1, self.image_size, 1) - p_x_z.mu.view(-1, self.image_size, 1))
            mu -= mu_prev.unsqueeze(-1)
            mu = torch.matmul(torch.inverse(precision), mu)
            mu = mu.squeeze(-1)
            mu += mu_prev
        else:
            raise ValueError

        return mu, precision

    def forward(self, x):
        q_z_x, _ = self.encoder.forward(x)
        mu = q_z_x.mu

        for i in range(self.n_update):
            p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
            mu_new, precision = self.solve_mu(x, mu, p_x_z, W_dec)
            lr = self.update_rate(i)
            mu = (1 - lr) * mu + lr * mu_new

        p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
        var_inv = torch.exp(-self.decoder.logvar).unsqueeze(1)
        precision = torch.matmul(W_dec.transpose(1, 2) * var_inv, W_dec)
        precision += torch.eye(precision.shape[1]).unsqueeze(0).cuda()

        # Update with analytically calulated mean and covariance.
        q_z_x = distribution.Gaussian(mu, precision)
        z = q_z_x.sample() # reparam trick
        p_x_z = self.decoder.forward(z)

        return self.loss(x, z, p_x_z, self.prior, q_z_x)

    def loss(self, x, z, p_x_z, p_z, q_z_x):
        return -torch.mean(p_x_z.log_probability(x.view(-1, self.image_size))
                           + p_z.log_probability(z)
                           - q_z_x.log_probability(z))

    def importance_sample(self, x):
        q_z_x, _ = self.encoder.forward(x)
        mu = q_z_x.mu

        for i in range(self.n_update):
            p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
            mu_new, precision = self.solve_mu(x, mu, p_x_z, W_dec)
            lr = self.update_rate(i)
            mu = (1 - lr) * mu + lr * mu_new

        p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
        var_inv = torch.exp(-self.decoder.logvar).unsqueeze(1)
        precision = torch.matmul(W_dec.transpose(1, 2) * var_inv, W_dec)
        precision += torch.eye(precision.shape[1]).unsqueeze(0).cuda()

        # Update with analytically calulated mean and covariance.
        q_z_x = distribution.Gaussian(mu, precision)
        q_z_x = q_z_x.repeat(n_importance_sample)
        z = q_z_x.sample()
        p_x_z = self.decoder(z)
        x = x.view(-1, self.image_size).unsqueeze(1).repeat(1, n_importance_sample, 1).view(-1, self.image_size)
        return self.importance_weighting(x, z, p_x_z, self.prior, q_z_x)

    def write_summary(self, x, writer, epoch):
        with torch.no_grad():
            q_z_x, _ = self.encoder.forward(x)
            mu = q_z_x.mu

            for i in range(self.n_update):
                p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
                writer.add_image(f'reconstruction_mu/{i}',
                                 vutils.make_grid(self.dataset.unpreprocess(p_x_z.mu).clamp(0, 1)),
                                 epoch)
                writer.add_scalar(f'recon_error/{i}',
                                -torch.mean(p_x_z.log_probability(x)).item(),
                                epoch)

                mu_new, precision = self.solve_mu(x, mu, p_x_z, W_dec)
                lr = self.update_rate(i)
                mu = (1 - lr) * mu + lr * mu_new

            p_x_z, W_dec = self.decoder.forward(mu, compute_jacobian=True)
            _, precision = self.solve_mu(x, mu, p_x_z, W_dec)
            writer.add_image(f'reconstruction_mu',
                             vutils.make_grid(self.dataset.unpreprocess(p_x_z.mu).clamp(0, 1)),
                             epoch)

            q_z_x = distribution.Gaussian(mu, precision)
            z = q_z_x.sample()
            p_x_z = self.decoder.forward(z)

            writer.add_scalar('kl_div',
                              torch.mean(-self.prior.log_probability(z)
                                         + q_z_x.log_probability(z)).item(),
                              epoch)
            writer.add_scalar('recon_error',
                              -torch.mean(p_x_z.log_probability(x)).item(),
                              epoch)
            writer.add_image('data',
                             vutils.make_grid(self.dataset.unpreprocess(x)),
                             epoch)
            writer.add_image('reconstruction_z',
                             vutils.make_grid(self.dataset.unpreprocess(p_x_z.mu).clamp(0, 1)),
                             epoch)

            sample = torch.randn(len(x), z.shape[1]).cuda()
            sample = self.decoder(sample).mu
            writer.add_image('generated',
                             vutils.make_grid(self.dataset.unpreprocess(sample).clamp(0, 1)),
                             epoch)


class HouseHolderVAE(VAE):
    def __init__(self, dataset, z_dim, output_dist, x_dim, enc_dim, dec_dim, n_flow, **kwargs):
        super().__init__(dataset, z_dim, output_dist, x_dim, enc_dim, dec_dim)
        self.householderflow = network.HouseHolderFlow(h_dim=enc_dim, z_dim=z_dim, n_flow=n_flow)

    def forward(self, x):
        q_z_x, h = self.encoder(x) # returns distribution
        H = self.householderflow(h)
        z = q_z_x.sample()
        z_flow = torch.matmul(H, z.unsqueeze(-1)).squeeze(-1)
        p_x_z = self.decoder(z_flow)
        return self.loss(x, z, z_flow, p_x_z, self.prior, q_z_x)

    def loss(self, x, z, z_flow, p_x_z, p_z, q_z_x):
        return -torch.mean(p_x_z.log_probability(x.view(-1, self.image_size))
                           + p_z.log_probability(z_flow)
                           - q_z_x.log_probability(z))

    def importance_weighting(self, x, z, z_flow, p_x_z, p_z, q_z_x):
        log_weights = (p_x_z.log_probability(x)
                       + p_z.log_probability(z_flow)
                       - q_z_x.log_probability(z)).view(-1, n_importance_sample)
        m = log_weights.max(1, keepdim=True)[0]
        weights = torch.exp(log_weights - m)
        loglikelihood = torch.mean(torch.log(weights.mean(dim=1)) + m, 1).sum()
        return loglikelihood

    def importance_sample(self, x):
        q_z_x, h = self.encoder(x) # returns distribution
        H = self.householderflow(h)
        H = H.unsqueeze(1).repeat(1, n_importance_sample, 1, 1).view(-1, *H.shape[1:])
        q_z_x = q_z_x.repeat(n_importance_sample)
        z = q_z_x.sample()
        z_flow = torch.matmul(H, z.unsqueeze(-1)).squeeze(-1)
        p_x_z = self.decoder(z_flow)
        x = x.view(-1, self.image_size).unsqueeze(1).repeat(1, n_importance_sample, 1).view(-1, self.image_size)
        return self.importance_weighting(x, z, z_flow, p_x_z, self.prior, q_z_x)

    def write_summary(self, x, writer, epoch):
        with torch.no_grad():
            q_z_x, h = self.encoder(x) # returns distribution
            H = self.householderflow(h)
            z = q_z_x.sample()
            z_flow = torch.matmul(H, z.unsqueeze(-1)).squeeze(-1)
            p_x_z = self.decoder(z_flow)

            writer.add_scalar('kl_div',
                              torch.mean(-self.prior.log_probability(z_flow)
                                         + q_z_x.log_probability(z)).item(),
                              epoch)
            writer.add_scalar('recon_error',
                              -torch.mean(p_x_z.log_probability(x)).item(),
                              epoch)
            writer.add_image('data',
                             vutils.make_grid(self.dataset.unpreprocess(x)),
                             epoch)
            writer.add_image('reconstruction_z',
                             vutils.make_grid(self.dataset.unpreprocess(p_x_z.mu).clamp(0, 1)),
                             epoch)

            sample = torch.randn(len(x), z.shape[1]).cuda()
            sample = self.decoder(sample).mu
            writer.add_image('generated',
                             vutils.make_grid(self.dataset.unpreprocess(sample).clamp(0, 1)),
                             epoch)


class SAVAE(VAE):
    def __init__(self, dataset, z_dim, output_dist, x_dim, enc_dim, dec_dim, svi_lr, n_svi_step, **kwargs):
        super().__init__(dataset, z_dim, output_dist, x_dim, enc_dim, dec_dim)
        self.svi_lr = svi_lr
        self.n_svi_step = n_svi_step

    def forward(self, x):
        with torch.enable_grad():
            q_z_x, _ = self.encoder.forward(x)
            mu_svi = q_z_x.mu
            logvar_svi = q_z_x.logvar

            for i in range(self.n_svi_step):
                q_z_x = distribution.DiagonalGaussian(mu_svi, logvar_svi)
                z = q_z_x.sample()
                p_x_z = self.decoder.forward(z)
                loss = self.loss(x, z, p_x_z, self.prior, q_z_x)
                # create_graph=True does this allow backprop through this when we update the whole thing
                mu_svi_grad, logvar_svi_grad = torch.autograd.grad(loss, inputs=(mu_svi, logvar_svi), create_graph=True)

                # mu_svi_grad = utils.clip_grad(mu_svi_grad, 1)
                # logvar_svi_grad = utils.clip_grad(logvar_svi_grad, 1)
                mu_svi_grad = utils.clip_grad_norm(mu_svi_grad, 5)
                logvar_svi_grad = utils.clip_grad_norm(logvar_svi_grad, 5)

                # gradient ascent.
                mu_svi = mu_svi + self.svi_lr * mu_svi_grad
                logvar_svi = logvar_svi + self.svi_lr * logvar_svi_grad

            # obtain z_K
            q_z_x = distribution.DiagonalGaussian(mu_svi, logvar_svi)
            z_K = q_z_x.sample()
            p_x_z = self.decoder.forward(z_K)

            loss = self.loss(x, z_K, p_x_z, self.prior, q_z_x)
            return loss

    def importance_sample(self, x):
        with torch.enable_grad():
            q_z_x, _ = self.encoder.forward(x)
            mu_svi = q_z_x.mu
            logvar_svi = q_z_x.logvar

            for i in range(self.n_svi_step):
                q_z_x = distribution.DiagonalGaussian(mu_svi, logvar_svi)
                z = q_z_x.sample()
                p_x_z = self.decoder.forward(z)
                loss = self.loss(x, z, p_x_z, self.prior, q_z_x)
                # create_graph=True does this allow backprop through this when we update the whole thing
                mu_svi_grad, logvar_svi_grad = torch.autograd.grad(loss, inputs=(mu_svi, logvar_svi), create_graph=True)

                # mu_svi_grad = utils.clip_grad(mu_svi_grad, 1)
                # logvar_svi_grad = utils.clip_grad(logvar_svi_grad, 1)
                mu_svi_grad = utils.clip_grad_norm(mu_svi_grad, 5)
                logvar_svi_grad = utils.clip_grad_norm(logvar_svi_grad, 5)
                # gradient ascent.
                mu_svi = mu_svi + self.svi_lr * mu_svi_grad
                logvar_svi = logvar_svi + self.svi_lr * logvar_svi_grad

            # obtain z_K
            q_z_x = distribution.DiagonalGaussian(mu_svi, logvar_svi)
            q_z_x = q_z_x.repeat(n_importance_sample)
            z = q_z_x.sample()
            p_x_z = self.decoder(z)
            x = x.view(-1, self.image_size).unsqueeze(1).repeat(1, n_importance_sample, 1).view(-1, self.image_size)

            return self.importance_weighting(x, z, p_x_z, self.prior, q_z_x)

    def write_summary(self, x, writer, epoch):
        q_z_x, _ = self.encoder.forward(x)
        mu_svi = q_z_x.mu
        logvar_svi = q_z_x.logvar

        for i in range(self.n_svi_step):
            q_z_x = distribution.DiagonalGaussian(mu_svi, logvar_svi)
            z = q_z_x.sample()
            p_x_z = self.decoder.forward(z)
            loss = self.loss(x, z, p_x_z, self.prior, q_z_x)
            # create_graph=True does this allow backprop through this when we update the whole thing
            mu_svi_grad, logvar_svi_grad = torch.autograd.grad(loss, inputs=(mu_svi, logvar_svi), create_graph=True)

            # mu_svi_grad = utils.clip_grad(mu_svi_grad, 1)
            # logvar_svi_grad = utils.clip_grad(logvar_svi_grad, 1)
            mu_svi_grad = utils.clip_grad_norm(mu_svi_grad, 5)
            logvar_svi_grad = utils.clip_grad_norm(logvar_svi_grad, 5)
            # gradient ascent.
            mu_svi = mu_svi + self.svi_lr * mu_svi_grad
            logvar_svi = logvar_svi + self.svi_lr * logvar_svi_grad

        # obtain z_K
        q_z_x = distribution.DiagonalGaussian(mu_svi, logvar_svi)
        z_K = q_z_x.sample()
        p_x_z = self.decoder.forward(z_K)

        writer.add_scalar('kl_div',
                          torch.mean(-self.prior.log_probability(z_K)
                                     + q_z_x.log_probability(z_K)).item(),
                          epoch)
        writer.add_scalar('recon_error',
                            -torch.mean(p_x_z.log_probability(x)).item(),
                            epoch)
        writer.add_image('data',
                            vutils.make_grid(self.dataset.unpreprocess(x)),
                            epoch)
        writer.add_image('reconstruction_z',
                            vutils.make_grid(self.dataset.unpreprocess(p_x_z.mu).clamp(0, 1)),
                            epoch)

        sample = torch.randn(len(x), z.shape[1]).cuda()
        sample = self.decoder(sample).mu
        writer.add_image('generated',
                            vutils.make_grid(self.dataset.unpreprocess(sample).clamp(0, 1)),
                            epoch)


class IAF(VAE):
    def __init__(self, dataset, z_dim, output_dist, x_dim, enc_dim, dec_dim, n_flow, iaf_dim, **kwargs):
        super().__init__(dataset, z_dim, output_dist, x_dim, enc_dim, dec_dim)
        self.n_flow = n_flow
        self.iaf_dim = iaf_dim
        self.AutoregressiveNN = nn.ModuleList(
            [network.MADE(self.z_dim, iaf_dim) for _ in range(n_flow)])

    def forward(self, x):
        q_z_x, h = self.encoder(x)
        z = q_z_x.sample()
        log_q_z_x = q_z_x.log_probability(z)
        z, log_q_z_x = self.iaf(z, h, log_q_z_x)
        p_x_z = self.decoder(z)

        return self.loss(x, z, p_x_z, self.prior, log_q_z_x)

    def iaf(self, z, h, log_density):
        self.reverse_indices = torch.LongTensor(list(range(self.z_dim))[::-1]).cuda()
        for i in range(self.n_flow):
            # Numerically stable reparametrization (inspired by LSTM)
            if i != 0:
                # Reverse the order of z dimensions
                z = torch.index_select(z, 1, self.reverse_indices)

            m, s = self.AutoregressiveNN[i](z, h)
            sigma = torch.sigmoid(s)
            z = sigma * z + (1 - sigma) * m

            # Determinant of a lower triangular matrix = product of diagonal terms
            # => log-determinant of Jacobian = - log sum(sigma)
            if log_density.size(0) > 1:
                log_density -= torch.log(sigma).sum(1)
            else:
                log_density -= torch.log(sigma).sum()

        # Put the order back
        if self.n_flow % 2 == 0:
            z = torch.index_select(z, 1, self.reverse_indices)

        return z, log_density

    def loss(self, x, z, p_x_z, p_z, log_q_z_x):
        return -torch.mean(p_x_z.log_probability(x.view(-1, self.image_size))
                           + p_z.log_probability(z)
                           - log_q_z_x)

    def importance_weighting(self, x, z, p_x_z, p_z, log_q_z_x):
        log_weights = (p_x_z.log_probability(x)
                       + p_z.log_probability(z)
                       - log_q_z_x).view(-1, n_importance_sample)
        m = log_weights.max(1, keepdim=True)[0]
        weights = torch.exp(log_weights - m)
        loglikelihood = torch.mean(torch.log(weights.mean(dim=1)) + m, 1).sum()
        return loglikelihood

    def importance_sample(self, x):
        q_z_x, h = self.encoder(x)
        q_z_x = q_z_x.repeat(n_importance_sample)
        h = h.repeat(n_importance_sample, 1)
        z = q_z_x.sample()
        log_q_z_x = q_z_x.log_probability(z)
        z, log_q_z_x = self.iaf(z, h, log_q_z_x)
        p_x_z = self.decoder(z)
        x = x.view(-1, self.image_size).unsqueeze(1).repeat(1, n_importance_sample, 1).view(-1, self.image_size)
        return self.importance_weighting(x, z, p_x_z, self.prior, log_q_z_x)

    def write_summary(self, x, writer, epoch):
        with torch.no_grad():
            q_z_x, h = self.encoder(x)
            z = q_z_x.sample()
            log_q_z_x = q_z_x.log_probability(z)
            z, log_q_z_x = self.iaf(z, h, log_q_z_x)
            p_x_z = self.decoder.forward(z)

            writer.add_scalar('kl_div',
                              torch.mean(-self.prior.log_probability(z)
                                         + log_q_z_x).item(),
                              epoch)
            writer.add_scalar('recon_error',
                              -torch.mean(p_x_z.log_probability(x)).item(),
                              epoch)
            writer.add_image('data',
                             vutils.make_grid(self.dataset.unpreprocess(x)),
                             epoch)
            writer.add_image('reconstruction_z',
                             vutils.make_grid(self.dataset.unpreprocess(p_x_z.mu).clamp(0, 1)),
                             epoch)

            sample = torch.randn(len(x), z.shape[1]).cuda()
            sample = self.decoder(sample).mu
            writer.add_image('generated',
                             vutils.make_grid(self.dataset.unpreprocess(sample).clamp(0, 1)),
                             epoch)
