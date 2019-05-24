from __future__ import print_function
import os
import math
import time
from pprint import pprint
from datetime import datetime
import torch
from torch import nn, optim
import numpy as np
from tensorboardX import SummaryWriter
import colorful
import datasets
import models


class Experiment():
    def __init__(self, model, dataset, logit_transform, batch_size, n_epochs,
                 log_interval, z_dim, output_dist, hidden_dim, learning_rate,
                 svi_lr, n_svi_step, n_update, update_lr, n_flow, iaf_dim,
                 weight_decay=0.0, seed=None, base_dir='./checkpoints'):
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.dataset = getattr(datasets, dataset)(batch_size=batch_size,
                                                  binarize=output_dist=="bernoulli",
                                                  logit_transform=logit_transform)
        self.model = getattr(models, model)(dataset=self.dataset,
                                            z_dim=z_dim,
                                            output_dist=output_dist,
                                            x_dim=self.dataset.dim,
                                            enc_dim=hidden_dim,
                                            dec_dim=hidden_dim,
                                            svi_lr=svi_lr,
                                            n_svi_step=n_svi_step,
                                            n_update=n_update,
                                            update_lr=update_lr,
                                            n_flow=n_flow, iaf_dim=iaf_dim).cuda()


        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.log_interval = log_interval
        self.z_dim = z_dim
        self.output_dist = output_dist
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.svi_lr = svi_lr
        self.n_svi_step = n_svi_step
        self.n_update = n_update
        self.update_lr = update_lr

        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(10000)

        torch.manual_seed(self.seed)

        self.name = (
            f'{self.timestamp}.logit_transform={logit_transform}'
            f'.out_dist={output_dist}.z_dim={z_dim}.hid_dim={hidden_dim}'
            f'.lr={learning_rate}.weight_decay={weight_decay}')
        if model == "LVAE":
            self.name += f'.n_update={n_update}.update_lr={update_lr}'
        elif model == "SAVAE":
            self.name += f'.svi_lr={svi_lr}.n_svi_step={n_svi_step}'
        elif model == "HF":
            self.name += f'.n_flow={n_flow}'
        elif model == "IAF":
            self.name += f'.n_flow={n_flow}.iaf_dim={iaf_dim}'

        self.save_dir = os.path.join(base_dir, dataset, model, self.name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate,
                                    weight_decay=weight_decay)

        self.epoch = 1
        self.best_epoch = None
        self.best_test_loss = None
        self.writer = SummaryWriter(self.save_dir)

        pprint(vars(self))
        with open(os.path.join(self.save_dir, 'log.txt'), 'w') as f:
            pprint(vars(self), f)

    def run(self):
        self.initialize_params()
        self.save_model(0)

        while self.epoch <= self.n_epochs:
            self.train_epoch()
            self.test()
            self.epoch += 1

        print(colorful.bold_green(f'\n====> Best Epoch: {self.best_epoch}').styled_string)
        best_checkpoint_path = os.path.join(self.save_dir, str(self.best_epoch) + '.pkl')
        self.importance_sample(best_checkpoint_path)
        self.writer.close()

    def initialize_params(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.pow(2, 1.0/3), mode='fan_out')

        if self.output_dist == 'gaussian':
            self.model.decoder.logvar.data = self.dataset.logvar

    def save_model(self, epoch):
        save_path = os.path.join(self.save_dir, str(epoch) + '.pkl')
        print(colorful.bold_yellow('Save model parameters to {}'.format(save_path)).styled_string)
        torch.save(self.model.state_dict(), save_path)

    def train_epoch(self):
        # set to train mode
        self.model.train()
        train_loss = 0
        train_loader = self.dataset.train_loader

        epoch_start_time = time.time()

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda()
            data = self.dataset.preprocess(data)

            if batch_idx == 0 and self.epoch == 1:
                self.model.write_summary(data, self.writer, 0)

            self.optimizer.zero_grad()
            loss = self.model(data)
            loss.backward()

            train_loss += loss.item() * len(data)
            assert not np.isnan(loss.item())

            self.optimizer.step()

            if self.log_interval is not None:
                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader),
                        loss.item()))

        duration = time.time() - epoch_start_time
        epoch_loss = train_loss / len(train_loader.dataset)
        print(colorful.bold_green('====> Epoch: {} Average loss: {:.4f} Duration(sec): {}'.format(self.epoch, epoch_loss, duration)).styled_string)
        self.writer.add_scalar('train/loss', epoch_loss, self.epoch)
        self.model.write_summary(data, self.writer, self.epoch)

    def test(self):
        self.model.eval()
        test_loss = 0
        test_loader = self.dataset.test_loader
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.cuda()
                data = self.dataset.preprocess(data)

                test_loss += self.model(data).item() * len(data)

        test_loss /= len(test_loader.dataset)
        if self.best_test_loss is None or test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            self.best_epoch = self.epoch
            self.save_model(self.epoch)

        print(colorful.bold_red('====> Test set loss: {:.4f}'.format(test_loss)).styled_string)
        self.writer.add_scalar('test/loss', test_loss, self.epoch)

    def importance_sample(self, checkpoint):
        self.model.load_state_dict(torch.load(checkpoint))
        self.model.eval()
        test_loglikelihood = 0
        test_loader = self.dataset.test_loader
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.cuda()
                data = self.dataset.preprocess(data)

                if models.n_importance_sample > 1000:
                    for data_ in data:
                        test_loglikelihood += self.model.importance_sample(data_.unsqueeze(0))
                else:
                    test_loglikelihood += self.model.importance_sample(data)

        test_loglikelihood /= len(test_loader.dataset)

        print(colorful.bold_green('====> Test set loglikelihood: {:.4f}'.format(test_loglikelihood)).styled_string)
        self.writer.add_scalar('test/loglikelihood', test_loglikelihood, self.best_epoch)
