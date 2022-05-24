import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset

from dataset import RawDataset
from utils import set_seed, split, fit


class Encoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, k_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim + k_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.linear_mu = nn.Linear(h_dim, z_dim)
        self.linear_logvar = nn.Linear(h_dim, z_dim)
        self.embed = nn.Linear(2, k_dim)

    def forward(self, x, k):
        x = torch.cat([x, self.embed(k)], dim=1)
        h = self.encoder(x)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, k_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + k_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, x_dim))
        self.embed = nn.Linear(2, k_dim)

    def forward(self, z, k):
        z = torch.cat([z, self.embed(k)], dim=1)
        return self.decoder(z)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def loss_vae(recon_x, x, mu, log_var):
    mse = F.mse_loss(recon_x, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return 10 * mse + 0.5 * kl


def forward_fn(encoder, decoder, x, k):
    x = x.to(device)
    k = k.to(device)
    z, mu, logvar = encoder(x, k)
    x_recon = decoder(z, k)
    loss = loss_vae(x_recon, x, mu, logvar)
    return x_recon, loss


def pretrain_fn(encoder,
                encoder_optimizer,
                decoders,
                decoder_optimizers,
                loaders):
    loaders = list(loaders.values())
    for x_list in zip(*loaders):
        for i, (x, k) in enumerate(x_list):
            _, loss = forward_fn(encoder, decoders[i], x, k)
            encoder_optimizer.zero_grad()
            decoder_optimizers[i].zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizers[i].step()


def metatrain_fn(encoder,
                 encoder_optimizer,
                 decoders,
                 decoder_optimizers,
                 loaders):
    loaders = list(loaders.values())
    for idx, x_list in enumerate(zip(*loaders)):
        # Phase 1: Fix enc, train different dec with only one batch
        if idx == 0:
            x_list_fixed = x_list
        for i, (x, k) in enumerate(x_list_fixed):
            _, loss = forward_fn(encoder, decoders[i], x, k)
            decoder_optimizers[i].zero_grad()
            loss.backward()
            decoder_optimizers[i].step()

        # Phase 2: Fix dec, train meta encoder
        for i, (x, k) in enumerate(x_list):
            _, loss = forward_fn(encoder, decoders[i], x, k)
            encoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()


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
            decoder_optimizers[i].zero_grad()
            loss.backward()
            decoder_optimizers[i].step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ----- Phase1: Pretrain -----
    # Dataset parameters
    parser.add_argument('--pretrain_dev_size', type=float, default=0.2)
    parser.add_argument('--pretrain_test_size', type=float, default=0.2)
    # Learning parameters
    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--pretrain_batch_size', type=int, default=10)
    parser.add_argument('--pretrain_lr', type=float, default=1e-2)
    # ----- Phase2: Metatrain -----
    # Dataset parameters
    parser.add_argument('--metatrain_dev_size', type=float, default=0.2)
    parser.add_argument('--metatrain_test_size', type=float, default=0.2)
    # Learning parameters
    parser.add_argument('--metatrain_epochs', type=int, default=100)
    parser.add_argument('--metatrain_batch_size', type=int, default=10)
    parser.add_argument('--metatrain_lr', type=float, default=1e-2)
    # ----- Phase3: Finetune -----
    # Dataset parameters
    parser.add_argument('--finetune_dev_size', type=float, default=0.2)
    parser.add_argument('--finetune_test_size', type=float, default=0.6)
    # Learning parameters
    parser.add_argument('--finetune_epochs', type=int, default=100)
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
        args.logdir, 'MetaCVAE_' + datetime.now().strftime('%Y%m%d%H%M%S'))
    set_seed(args.seed)
    os.makedirs(args.logdir)

    dataset1 = RawDataset('./data/水泵聯軸器_20211228.csv')
    dataset2 = RawDataset('./data/水泵馬達_20211116.csv')
    pretrain_size = int(len(dataset1) * 0.5)
    indices = list(torch.randperm(len(dataset1)))
    pretrain_indices = indices[:pretrain_size]
    metatrain_indices = indices[pretrain_size:]

    pretrain_dataset = Subset(dataset1, pretrain_indices)
    metatrain_dataset = Subset(dataset1, metatrain_indices)
    finetune_dataset = dataset2

    # Split pretrain dataset
    pretrain_splits = split(
        pretrain_dataset, args.pretrain_dev_size, args.pretrain_test_size)
    # Split metatrain dataset
    metatrain_splits = split(
        metatrain_dataset, args.metatrain_dev_size, args.metatrain_test_size)
    # Split finetune dataset
    finetune_splits = split(
        finetune_dataset, args.finetune_dev_size, args.finetune_test_size)

    start_epoch = 0
    logdir_1 = fit(
        train_fn=pretrain_fn,
        forward_fn=forward_fn,
        loss_name="ELBO",
        datasets_dict={
            "pretrain": pretrain_splits,
        },
        target_dataset='pretrain',
        epochs=list(range(start_epoch, start_epoch + args.pretrain_epochs)),
        batch_size=args.pretrain_batch_size,
        lr=args.pretrain_lr,
        root_logdir=args.logdir,
        num_workers=args.num_workers,
        Encoder=Encoder,
        Decoder=Decoder,
        x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim, k_dim=args.k_dim,
        encoder_path=None,
        decoder_path=None,)

    start_epoch = args.pretrain_epochs
    logdir_2 = fit(
        train_fn=metatrain_fn,
        forward_fn=forward_fn,
        loss_name="ELBO",
        datasets_dict={
            "pretrain": pretrain_splits,
            "metatrain": metatrain_splits,
        },
        target_dataset='metatrain',
        epochs=list(range(start_epoch, start_epoch + args.metatrain_epochs)),
        batch_size=args.metatrain_batch_size,
        lr=args.metatrain_lr,
        root_logdir=args.logdir,
        num_workers=args.num_workers,
        Encoder=Encoder,
        Decoder=Decoder,
        x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim, k_dim=args.k_dim,
        encoder_path=logdir_1,
        decoder_path=logdir_1,)

    start_epoch = args.pretrain_epochs + args.metatrain_epochs
    logdir_3 = fit(
        train_fn=finetune_fn,
        forward_fn=forward_fn,
        loss_name="ELBO",
        datasets_dict={
            "pretrain": pretrain_splits,
            "metatrain": metatrain_splits,
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
        encoder_path=logdir_2,
        decoder_path=logdir_1,)
