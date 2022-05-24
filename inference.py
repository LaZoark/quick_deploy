import argparse
import random

import torch
import pandas as pd

import train_vanilla_ae as vanilla_ae
import train_meta_ae as meta_ae
import train_meta_cvae as meta_cvae
from dataset import RawDataset


Encoders = {
    'vanilla_ae': vanilla_ae.Encoder,
    'meta_ae': meta_ae.Encoder,
    'meta_cvae': meta_cvae.Encoder,
}

Decoders = {
    'vanilla_ae': vanilla_ae.Decoder,
    'meta_ae': meta_ae.Decoder,
    'meta_cvae': meta_cvae.Decoder,
}

forward_fns = {
    'vanilla_ae': vanilla_ae.forward_fn,
    'meta_ae': meta_ae.forward_fn,
    'meta_cvae': meta_cvae.forward_fn,
}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        "This is an inference example which randomly tests the reconstruction"
        "MSE of a single row in finetune set")
    parser.add_argument('--model', default='meta_cvae',
                        choices=Encoders.keys(),
                        help="model type")
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the weights (*.pt) file')
    # model architecture
    parser.add_argument('--x_dim', type=int, default=24)
    parser.add_argument('--h_dim', type=int, default=10)
    parser.add_argument('--z_dim', type=int, default=5)
    parser.add_argument('--k_dim', type=int, default=2)
    args = parser.parse_args()

    encoder = Encoders[args.model](
        x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim, k_dim=args.k_dim)
    encoder = encoder.to(device)
    encoder.eval()
    decoder = Decoders[args.model](
        x_dim=args.x_dim, h_dim=args.h_dim, z_dim=args.z_dim, k_dim=args.k_dim)
    decoder = decoder.to(device)
    decoder.eval()

    ckpt = torch.load(args.weights, map_location='cpu')
    encoder.load_state_dict(ckpt['encoder'])
    decoder.load_state_dict(ckpt['decoder_finetune'])
    mean = ckpt['feature_mean'].to(device)
    std = ckpt['feature_std'].to(device)

    df = pd.read_csv('./data/水泵馬達_20211116.csv')
    df = df[RawDataset.feature_columns + RawDataset.conditional_columns]
    rand_idx = random.randint(0, len(df) - 1)
    feature = torch.tensor(df.iloc[rand_idx, :].values).float().to(device)
    feature = (feature - mean) / std
    x = feature[:len(RawDataset.feature_columns)]
    k = feature[len(RawDataset.feature_columns):]

    x = x.unsqueeze(0)
    k = k.unsqueeze(0)
    x_recon, _ = forward_fns[args.model](encoder, decoder, x, k)
    mse = (x - x_recon).pow(2).sum()
    print(mse.item())
