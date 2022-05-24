import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import RawDataset
from utils import set_seed, split, fit


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, k_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim),
            nn.ReLU())

    def forward(self, x, k):
        x = torch.cat([x, k], dim=1)
        z = self.encoder(x)
        return z


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, k_dim):
        super().__init__()
        self.x_dim = x_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim + 2))

    def forward(self, z):
        h = self.decoder(z)
        x, k = h.split([self.x_dim, 2], dim=1)
        return x, k


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def loss_ae(recon_x, recon_k, x, k):
    return (
        F.mse_loss(recon_x, x, reduction='sum') +
        F.mse_loss(recon_k, k, reduction='sum')
    )


def forward_fn(encoder, decoder, x, k):
    x = x.to(device)
    k = k.to(device)
    z = encoder(x, k)
    x_recon, recon_k = decoder(z)
    loss = loss_ae(x_recon, recon_k, x, k)
    return x_recon, loss


def finetune_fn(encoder,
                encoder_optimizer,
                decoders,
                decoder_optimizers,
                loaders):
    loaders = list(loaders.values())
    for x_list in zip(*loaders):
        # only fit decoder
        for i, (x, k) in enumerate(x_list):
            _, loss = forward_fn(encoder, decoders[i], x, k)
            encoder_optimizer.zero_grad()
            decoder_optimizers[i].zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizers[i].step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ----- Phase1: Pretrain -----
    # Dataset parameters
    parser.add_argument('--finetune_dev_size', type=float, default=0.2)
    parser.add_argument('--finetune_test_size', type=float, default=0.6)
    # Learning parameters
    parser.add_argument('--finetune_epochs', type=int, default=200)
    parser.add_argument('--finetune_batch_size', type=int, default=10)
    parser.add_argument('--finetune_lr', type=float, default=1e-2)
    # ----- Common -----
    # Model parameters
    parser.add_argument('--x_dim', type=int, default=24)
    parser.add_argument('--h_dim', type=int, default=10)
    parser.add_argument('--z_dim', type=int, default=5)
    parser.add_argument('--k_dim', type=int, default=2)
    # Dataset parameters
    parser.add_argument('--num_workers', type=int, default=3)
    # Logging parameters
    parser.add_argument('--logdir', type=str, default='./logs')
    # Seed
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    args.logdir = os.path.join(
        args.logdir, 'VanillaAE_' + datetime.now().strftime('%Y%m%d%H%M%S'))
    set_seed(args.seed)
    os.makedirs(args.logdir)

    finetune_splits = RawDataset('./data/水泵馬達_20211116.csv')

    # Split pretrain dataset
    finetune_splits = split(
        finetune_splits, args.finetune_dev_size, args.finetune_test_size)

    start_epoch = 0
    logdir_3 = fit(
        train_fn=finetune_fn,
        forward_fn=forward_fn,
        loss_name="MSE",
        datasets_dict={
            "finetune": finetune_splits,
        },
        target_dataset='finetune',
        epochs=list(range(start_epoch, start_epoch + args.finetune_epochs)),
        batch_size=args.finetune_batch_size,
        lr=args.finetune_lr,
        root_logdir=args.logdir,
        num_workers=args.num_workers,
        Encoder=Encoder,
        Decoder=Decoder,
        x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim, k_dim=args.k_dim,
        encoder_path=None,
        decoder_path=None,)
