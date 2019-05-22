from __future__ import print_function
import argparse
from experiment import *


parser = argparse.ArgumentParser(description='VLAE')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='path to the model checkpoint')
parser.add_argument('--model', type=str, default='VAE',
                    help='[VAE, VLAE, SAVAE, HF, IAF]')
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='[SVHN, MNIST, OMNIGLOT, FashionMNIST, CIFAR10]')
parser.add_argument('--logit_transform', type=bool, default=False,
                    help='wheter to apply logit transform to data for \
                    continuous output distributions')
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--n_epochs', type=int, default=2000,
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--base_dir', type=str, default='./checkpoints/',
                    help='(relative) base dir')

parser.add_argument('--z_dim', type=int, default=50,
                    help='latent space dimension')
parser.add_argument('--output_dist', type=str, default='gaussian',
                    help='One of [gaussian, bernoulli]')
parser.add_argument('--hidden_dim', type=int, default=500,
                    help='hidden unit dimension for encoder and decoder')
parser.add_argument('--learning_rate', type=float, default=5e-4,
                    help='Learning rate for ADAM optimizer')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='L2 weight decay')

# SAVAE parameters
parser.add_argument('--svi_lr', type=float, default=5e-4,
                    help='SVI lr. MNIST:0.1, CIFAR10:1e-3, ')
parser.add_argument('--n_svi_step', type=int, default=4,
                    help='SVI number of steps')

# VLAE parameters
parser.add_argument('--n_update', type=int, default=4,
                    help='number of updates')
parser.add_argument('--update_lr', type=float, default=0.5,
                    help='update learning rate')

# HouseholderFlow/IAF parameters
parser.add_argument('--n_flow', type=int, default=4,
                    help='number of householder flows to apply')
parser.add_argument('--iaf_dim', type=int, default=500,
                    help='dim for iaf layers')

args = parser.parse_args()


if __name__ == "__main__":
    exp = Experiment(model=args.model,
                     dataset=args.dataset,
                     logit_transform=args.logit_transform,
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
                     iaf_dim=args.iaf_dim,
                     weight_decay=args.weight_decay)

    exp.importance_sample(args.checkpoint)
