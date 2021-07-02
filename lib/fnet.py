#!/usr/bin/env python

# Code from https://github.com/rishikksh20/FNet-pytorch

import torch


class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, dim),
            torch.nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        return self.net(x)


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FNetBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rfft(x, signal_ndim=2, onesided=False)[..., 0]


class FNet(torch.nn.Module):
    def __init__(self, dim, depth, num_ch=19, seq_len=512, dropout=0.5):
        super().__init__()
        self.preproc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=dim,
                            kernel_size=(num_ch, 1), stride=(1, 1),
                            padding=(0, 0), bias=False),
            torch.nn.BatchNorm2d(num_features=dim, track_running_stats=False),
            torch.nn.GELU())
        self.layers = torch.nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(torch.nn.ModuleList([
                PreNorm(dim, FNetBlock()),
                PreNorm(dim, FeedForward(dim, dim, dropout=dropout))
            ]))
        self.predictor = torch.nn.Sequential(
            torch.nn.BatchNorm1d(num_features=dim),
            torch.nn.Linear(in_features=dim, out_features=2),
            torch.nn.ReLU())

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.preproc(x).squeeze(2)
        x = torch.cat((x, torch.ones(size=(*x.size()[:2], 1), dtype=x.dtype, device=x.device)), dim=2)
        x = x.permute(0, 2, 1)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = self.predictor(x[:, -1, :])
        return x
