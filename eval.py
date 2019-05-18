from __future__ import print_function
import argparse
from experiment import *


parser = argparse.ArgumentParser(description='VLAE')
parser.add_argument('--checkpoint', type=str, default=None, metavar='S',
                    help='model name')
parser.add_argument('--model', type=str, default='VAE', metavar='S',
                    help='model name')
parser.add_argument('--dataset', type=str, default='MNIST', metavar='S',
                    help='[SVHN, MNIST, OMNIGLOT, FashionMNIST, CIFAR10]')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--n_epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=None, metavar='N',
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--base_dir', type=str, default='./checkpoints/', metavar='S',
                    help='(relative) base dir')

parser.add_argument('--z_dim', type=int, default=50, metavar='N',
                    help='latent space dimension')
parser.add_argument('--output_dist', type=str, default='bernoulli', metavar='S',
                    help='One of [gaussian, bernoulli]')
parser.add_argument('--hidden_dim', type=int, default=500, metavar='N',
                    help='hidden unit dimension for encoder and decoder')
parser.add_argument('--learning_rate', type=float, default=5e-4, metavar='F',
                    help='Learning rate for ADAM optimizer')

# SAVAE parameters
parser.add_argument('--svi_lr', type=float, default=5e-4, metavar='F',
                    help='SVI lr. MNIST:0.1, CIFAR10:1e-3, ')
parser.add_argument('--n_svi_step', type=int, default=4, metavar='F',
                    help='SVI number of steps')

# VLAE parameters
parser.add_argument('--n_update', type=int, default=1, metavar='N',
                    help='number of updates')
parser.add_argument('--update_lr', type=float, default=0.5, metavar='N',
                    help='update learning rate')

# HouseholderFlow parameters
parser.add_argument('--n_flow', type=int, default=1, metavar='N',
                    help='number of householder flows to apply')
parser.add_argument('--iaf_dim', type=int, default=500, metavar='F',
                    help='dim for iaf layers')

args = parser.parse_args()


if __name__ == "__main__":
    exp = Experiment(model=args.model,
                     dataset=args.dataset,
                     batch_size=args.batch_size,
                     n_epochs=args.n_epochs,
                     log_interval=args.log_interval,
                     z_dim=args.z_dim,
                     output_dist=args.output_dist,
                     hidden_dim=args.hidden_dim,
                     learning_rate=args.learning_rate,
                     svi_lr=args.svi_lr,
                     n_svi_step=args.n_svi_step,
                     n_update=args.n_update,
                     update_lr=args.update_lr,
                     n_flow=args.n_flow,
                     seed=args.seed,
                     base_dir=args.base_dir,
                     iaf_dim=args.iaf_dim)

    exp.importance_sample(args.checkpoint)
