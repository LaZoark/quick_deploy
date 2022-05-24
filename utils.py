import os
import json
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from dataset import NormalizedDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # reproducable.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split(dataset, dev_size, test_size):
    """split dataset into train, dev and test sets."""
    dev_size = int(len(dataset) * dev_size)
    test_size = int(len(dataset) * test_size)
    train_size = len(dataset) - dev_size - test_size
    train, dev, test = random_split(
        dataset,
        lengths=[train_size, dev_size, test_size],
        generator=torch.Generator().manual_seed(0))
    mean, std = NormalizedDataset.calculate_statistic(train)
    train = NormalizedDataset(train, mean, std)
    dev = NormalizedDataset(dev, mean, std)
    test = NormalizedDataset(test, mean, std)
    return {
        "train": train,
        "dev": dev,
        "test": test,
    }


def loss_recon(recon_x, x):
    return F.mse_loss(recon_x, x, reduction='sum')


def test(encoder, decoders, forward_fn, loss_name, loaders_dict, split_name):
    encoder.eval()
    loss_dict = dict()
    for decoder, (name, loader) in zip(decoders, loaders_dict.items()):
        decoder.eval()
        loss_sum = 0
        loss_recon_sum = 0
        with torch.no_grad():
            for x, k in loader:
                x_recon, loss = forward_fn(encoder, decoder, x, k)
                loss_sum += loss.cpu().item()
                loss_recon_sum += loss_recon(x_recon.cpu(), x).item()
        decoder.train()
        count = len(loader.dataset)
        loss_dict[f"{name}/{split_name}/{loss_name}"] = loss_sum / count
        loss_dict[f"{name}/{split_name}/recon/MSE"] = loss_recon_sum / count
    encoder.train()
    return loss_dict


def fit(train_fn,
        forward_fn,
        loss_name,
        datasets_dict,
        target_dataset,
        epochs,
        batch_size,
        lr,
        root_logdir,
        num_workers,
        Encoder,
        Decoder,
        x_dim, h_dim, z_dim, k_dim,
        encoder_path=None,
        decoder_path=None):
    logdir = os.path.join(root_logdir, target_dataset)
    writer = SummaryWriter(logdir)

    loaders_dict = {
        "train": dict(),
        "dev": dict(),
        "test": dict(),
    }
    for dataset_name, dataset_split in datasets_dict.items():
        for split_name, dataset in dataset_split.items():
            loaders_dict[split_name][dataset_name] = DataLoader(
                dataset=dataset, batch_size=batch_size,
                shuffle=split_name == "train", num_workers=num_workers)

    # Load encoder if `encoder_path` is not None
    encoder = Encoder(x_dim, h_dim, z_dim, k_dim)
    encoder = encoder.to(device)
    if encoder_path:
        encoder_ckpt = torch.load(os.path.join(encoder_path, 'model.pt'))
        encoder.load_state_dict(encoder_ckpt['encoder'])

    # Load decoder if `decoder_path` is not None
    decoder = Decoder(x_dim, h_dim, z_dim, k_dim)
    decoder = decoder.to(device)
    if decoder_path:
        decoder_ckpt = torch.load(os.path.join(decoder_path, 'model.pt'))
        decoder.load_state_dict(decoder_ckpt['decoder'])

    # optimizer
    encoder_optimizer = Adam(encoder.parameters(), lr=lr)
    # learning rate linear decay to zero
    encoder_scheduler = OneCycleLR(
        encoder_optimizer, max_lr=lr, total_steps=len(epochs),
        pct_start=1e-10, anneal_strategy='linear')
    decoders = [deepcopy(decoder) for _ in range(len(datasets_dict))]
    # optimizer
    decoder_optimizers = [
        Adam(decoder.parameters(), lr=lr) for decoder in decoders]
    # learning rate linear decay to zero
    decoder_schedulers = [
        OneCycleLR(
            optimizer, max_lr=lr, total_steps=len(epochs),
            pct_start=1e-10, anneal_strategy='linear')
        for optimizer in decoder_optimizers
    ]

    best_dev_MSE = float('inf')
    target_key = f'{target_dataset}/dev/recon/MSE'
    pbar = tqdm(epochs)
    for epoch in pbar:
        train_fn(encoder,
                 encoder_optimizer,
                 decoders,
                 decoder_optimizers,
                 loaders_dict['train'])
        encoder_scheduler.step()
        for scheduler in decoder_schedulers:
            scheduler.step()
        # print message
        metrics = dict()
        for split_name in ['train', 'dev', 'test']:
            loss_dict = test(
                encoder, decoders, forward_fn, loss_name,
                loaders_dict[split_name], split_name)
            metrics.update(loss_dict)
        pbar.write(
            f'Epoch {epoch:3d} [{loss_name}] ' +
            ', '.join([
                f"{split_name}: "
                f"{metrics[f'{target_dataset}/{split_name}/{loss_name}']:5.2f}"
                for split_name in ['train', 'dev', 'test']]) +
            ' | [Recon] ' +
            ', '.join([
                f"{split_name}: "
                f"{metrics[f'{target_dataset}/{split_name}/recon/MSE']:5.2f}"
                for split_name in ['train', 'dev', 'test']]))
        # wrtie to tensorboard
        for k, v in metrics.items():
            writer.add_scalar(k, v, epoch)
        # save best model
        if metrics[target_key] < best_dev_MSE:
            best_dev_MSE = metrics[target_key]
            state_dict = {
                'epoch': epoch,
                'metrics': metrics,
                'encoder': encoder.state_dict(),
                'decoder': decoders[0].state_dict(),
                'feature_mean': datasets_dict[target_dataset]['train'].mean,
                'feature_std': datasets_dict[target_dataset]['train'].std,
            }
            for decoder, dataset_name in zip(decoders, datasets_dict.keys()):
                state_dict[f'decoder_{dataset_name}'] = decoder.state_dict()
            torch.save(state_dict, os.path.join(logdir, 'model.pt'))
    # best metrics
    metrics = state_dict['metrics']
    pbar.write(
        f'BEST      [{loss_name}] ' +
        ', '.join([
            f"{split_name}: "
            f"{metrics[f'{target_dataset}/{split_name}/{loss_name}']:5.2f}"
            for split_name in ['train', 'dev', 'test']]) +
        ' | [Recon] ' +
        ', '.join([
            f"{split_name}: "
            f"{metrics[f'{target_dataset}/{split_name}/recon/MSE']:5.2f}"
            for split_name in ['train', 'dev', 'test']]))
    # save best metrics to json file
    metrics_path = os.path.join(logdir, 'best_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            best_metrics = json.load(f)
        metrics.update(best_metrics)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4, sort_keys=True)
    writer.close()
    return logdir
