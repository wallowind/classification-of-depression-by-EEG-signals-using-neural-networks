#!/usr/bin/env python
import torch


class CNN_Zero(torch.nn.Module):
    def __init__(self, kernel_size=(19, 13), stride=6, **kwargs):
        super(CNN_Zero, self).__init__()
        # Could be Conv1d as well
        self.only_layer = torch.nn.Conv2d(
            in_channels=1, out_channels=2,
            kernel_size=kernel_size,
            stride=(1, stride))
        self.bn = torch.nn.BatchNorm2d(num_features=2, track_running_stats=False)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)  # Add virtual "channels" dimension
        out = self.act(self.bn(self.only_layer(x)))  # Verbose version of torch.nn.Sequential
        return out.mean(dim=-1).view(out.size(0), -1)


class CNN_First(torch.nn.Module):
    def __init__(self, features=32, kernel_size=(19, 13), stride=6, **kwargs):
        super(CNN_First, self).__init__()
        self.first_layer = torch.nn.Conv2d(
            in_channels=1, out_channels=features,
            kernel_size=kernel_size,
            stride=(1, stride))
        self.last_layer = torch.nn.Conv1d(
            in_channels=features, out_channels=2,
            kernel_size=1, stride=1)
        self.bn = torch.nn.BatchNorm2d(num_features=features, track_running_stats=False)
        self.act = torch.nn.ELU()

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        features = self.act(self.bn(self.first_layer(x)))
        out = self.act(self.last_layer(features.squeeze(2)))
        return out.mean(dim=-1).view(out.size(0), -1)


class CNN_Second(torch.nn.Module):
    def __init__(self, features=(32, 64), kernel_sizes=[(1, 13), (19, 6)],
                 strides=(6, 2), **kwargs):
        super(CNN_Second, self).__init__()
        self.first_layer = torch.nn.Conv2d(
            in_channels=1, out_channels=features[0],
            kernel_size=kernel_sizes[0], stride=(1, strides[0]))
        self.second_layer = torch.nn.Conv2d(
            in_channels=features[0], out_channels=features[1],
            kernel_size=kernel_sizes[1], stride=(1, strides[1]))
        self.last_layer = torch.nn.Conv1d(
            in_channels=features[1], out_channels=2,
            kernel_size=1, stride=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=features[0], track_running_stats=False)
        self.bn2 = torch.nn.BatchNorm2d(num_features=features[1], track_running_stats=False)
        self.act1 = torch.nn.GELU()
        self.act2 = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        simple_features = self.act1(self.bn1(self.first_layer(x)))
        complex_features = self.act1(self.bn2(self.second_layer(simple_features)))
        out = self.act2(self.last_layer(complex_features.squeeze(2)))
        return out.mean(dim=-1).view(out.size(0), -1)
