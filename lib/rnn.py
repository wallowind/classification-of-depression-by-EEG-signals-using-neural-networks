#!/usr/bin/env python
import torch
from .cnn import CNN_Second
from .rim import RIM


class RNN(CNN_Second):
    def __init__(self, features=(32, 64), kernel_sizes=[(1, 64), (19, 16)], strides=(8, 2), hidden_size=128):
        super(RNN, self).__init__(features, kernel_sizes=kernel_sizes, strides=strides)
        self.rnn_layer = torch.nn.GRU(
            input_size=features[1],
            hidden_size=hidden_size,
            batch_first=True)
        self.prediction_layer = torch.nn.Linear(
            in_features=hidden_size,
            out_features=2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=hidden_size, track_running_stats=False)

    def forward(self, x: torch.Tensor):
        # More info in: https://discuss.pytorch.org/t/why-do-we-need-flatten-parameters-when-using-rnn-with-dataparallel/46506/2
        self.rnn_layer.flatten_parameters()
        x = x.unsqueeze(1)
        simple_features = self.act1(self.bn1(self.first_layer(x)))
        complex_features = self.act1(self.bn2(self.second_layer(simple_features)))
        features = complex_features.squeeze(2).permute(0, 2, 1)
        _, rnn_out = self.rnn_layer(features)
        out = self.act2(self.prediction_layer(self.bn3(rnn_out.squeeze(0))))
        return out


class RNN_RIM(CNN_Second):
    def __init__(self, features=(32, 64), kernel_sizes=[(1, 64), (19, 16)], strides=(8, 2), rim_num=16, rim_hid=16):
        super(RNN_RIM, self).__init__(features, kernel_sizes=kernel_sizes, strides=strides)
        self.rim_layer = RIM(hid=rim_hid, rims=rim_num, sql=21, chl=features[1], top=4)
        self.prediction_layer = torch.nn.Linear(
            in_features=rim_num * rim_hid,
            out_features=2)
        self.bn3 = torch.nn.BatchNorm1d(num_features=rim_num * rim_hid, track_running_stats=False)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        simple_features = self.act1(self.bn1(self.first_layer(x)))
        complex_features = self.act1(self.bn2(self.second_layer(simple_features)))
        rim = self.rim_layer(complex_features.squeeze(2))
        out = self.act2(self.prediction_layer(self.bn3(rim.view(*rim.size()[:2], -1)[-1])))
        return out
