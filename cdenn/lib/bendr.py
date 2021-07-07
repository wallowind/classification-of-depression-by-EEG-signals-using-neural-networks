#!/usr/bin/env python

# Code and pretrainded weights from: https://github.com/SPOClab-ca/BENDR

import copy
import torch

WEIGHTS_PATH = "/home/viktor/Documents/umnik/report/bendr/"


class _Hax(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Permute(torch.nn.Module):
    def __init__(self, axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return x.permute(self.axes)


class Contextualizer(torch.nn.Module):
    def __init__(self, in_features, hidden_feedforward=3076, heads=8, layers=8,
                 dropout=0.15, activation='gelu', position_encoder=25):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self._transformer_dim = in_features * 3
        encoder = torch.nn.TransformerEncoderLayer(d_model=in_features * 3,
                                                   nhead=heads, dim_feedforward=hidden_feedforward,
                                                   dropout=dropout, activation=activation)
        encoder.norm1 = _Hax()
        encoder.norm2 = _Hax()
        self.norm = torch.nn.LayerNorm(self._transformer_dim)
        self.transformer_layers = torch.nn.ModuleList([copy.deepcopy(encoder) for _ in range(layers)])
        self.mask_replacement = torch.nn.Parameter(torch.normal(0, in_features**(-0.5), size=(in_features,)))
        self.position_encoder = position_encoder > 0
        if position_encoder:
            conv = torch.nn.Conv1d(in_features, in_features, position_encoder, padding=position_encoder // 2, groups=16)
            torch.nn.init.normal_(conv.weight, mean=0, std=2 / self._transformer_dim)
            torch.nn.init.constant_(conv.bias, 0)
            conv = torch.nn.utils.weight_norm(conv, dim=2)
            self.relative_position = torch.nn.Sequential(conv, torch.nn.GELU())
        self.input_conditioning = torch.nn.Sequential(
            Permute([0, 2, 1]),
            torch.nn.LayerNorm(in_features),
            torch.nn.Dropout(dropout),
            Permute([0, 2, 1]),
            torch.nn.Conv1d(in_features, self._transformer_dim, 1),
            Permute([2, 0, 1]),
        )
        self.output_layer = torch.nn.Conv1d(self._transformer_dim, in_features, 1)
        self.apply(self.init_bert_params)

    def init_bert_params(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
            # Tfixup
            module.weight.data = 0.67 * len(self.transformer_layers) ** (-0.25) * module.weight.data

    def freeze_features(self, unfreeze=False, finetuning=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def forward(self, x):
        if self.position_encoder:
            x = x + self.relative_position(x)
        x = self.input_conditioning(x)
        for layer in self.transformer_layers:
            x = layer(x)
        return self.output_layer(x.permute([1, 2, 0]))


class Encoder(torch.nn.Module):
    def __init__(self, in_features, encoder_h=256, enc_width=(3, 3, 3, 3, 3, 3),
                 dropout=0., projection_head=False, enc_downsample=(3, 2, 2, 2, 2, 2)):
        super().__init__()
        self.in_features = in_features
        self.encoder_h = encoder_h
        self._downsampling = enc_downsample
        self._width = enc_width
        self.encoder = torch.nn.Sequential()
        for i, (width, downsample) in enumerate(zip(enc_width, enc_downsample)):
            self.encoder.add_module("Encoder_{}".format(i), torch.nn.Sequential(
                torch.nn.Conv1d(in_features, encoder_h, width, stride=downsample, padding=width // 2),
                torch.nn.Dropout2d(dropout),
                torch.nn.GroupNorm(encoder_h // 2, encoder_h),
                torch.nn.GELU(),
            ))
            in_features = encoder_h
        if projection_head:
            self.encoder.add_module("projection-1", torch.nn.Sequential(
                torch.nn.Conv1d(in_features, in_features, 1),
                torch.nn.Dropout2d(dropout * 2),
                torch.nn.GroupNorm(in_features // 2, in_features),
                torch.nn.GELU()
            ))

    def downsampling_factor(self, samples):
        ceil = lambda x, y: x // y + 1 if x % y else x // y
        for factor in self._downsampling:
            samples = ceil(samples, factor)
        return samples

    def load(self, filename, strict=True):
        state_dict = torch.load(filename)
        self.load_state_dict(state_dict, strict=strict)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def freeze_features(self, unfreeze=False):
        for param in self.parameters():
            param.requires_grad = unfreeze

    def forward(self, x):
        return self.encoder(x)


class Bendr(torch.nn.Module):
    def __init__(self, n_hid=512, **kwargs):
        super(Bendr, self).__init__()
        self.encoder = Encoder(19, encoder_h=n_hid)
        self.contextualizer = Contextualizer(in_features=n_hid)
        self.ensemble = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.BatchNorm1d(num_features=n_hid),
                torch.nn.Linear(in_features=n_hid, out_features=2, bias=False),
                torch.nn.GELU())
            for _ in range(6)])
        self.load(path=WEIGHTS_PATH)

    def forward(self, x: torch.Tensor):
        x = self.contextualizer(self.encoder(x))
        y = torch.zeros(size=(6, x.size(0), 2), dtype=x.dtype, device=x.device)
        for i, m in enumerate(self.ensemble):
            y[i] += m(x[..., i])
        return y.mean(dim=0)

    def save(self, path):
        raise NotImplementedError("Safeguard for preventing rewriting of original Bendr.")
        self.encoder.save(path + "encoder.pt")
        self.contextualizer.save(path + "contextualizer.pt")
        torch.save(self.predictor.state_dict(), path + "predictor.pt")

    def load(self, path, strict=True, finetuning=True):
        # self.encoder.load(path + "encoder.pt", strict=strict)
        self.contextualizer.load(path + "contextualizer.pt", strict=strict)
        # self.encoder.freeze_features()
        # self.contextualizer.freeze_features()
