#!/usr/bin/env python

# Code is based on: https://github.com/dido1998/Recurrent-Independent-Mechanisms

import torch
import collections


class NoGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mask):
        ctx.mask = mask
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.mask, ctx.mask * 0


class RIM(torch.nn.Module):
    def __init__(self, sql=512, chl=19, rims=8, hid=128, top=3):
        super(RIM, self).__init__()
        self.cells = torch.nn.ModuleList(
            (torch.nn.GRU(input_size=chl + 1,
                          hidden_size=hid,
                          bias=False)
             for _ in range(rims)))
        self.Qir = torch.nn.Linear(in_features=hid, out_features=hid, bias=True)
        self.Kir = torch.nn.Linear(in_features=sql, out_features=hid, bias=False)
        self.Vir = torch.nn.Linear(in_features=sql, out_features=hid, bias=False)
        self.Qrr = torch.nn.Linear(in_features=hid, out_features=hid, bias=False)
        self.Krr = torch.nn.Linear(in_features=hid, out_features=hid, bias=False)
        self.Vrr = torch.nn.Linear(in_features=hid, out_features=hid, bias=False)
        self._scale = hid ** 0.5
        self._top = top

    def rim_competition_other(self, att_scores: torch.Tensor):
        # [sql, bs, rims, chls]
        i = att_scores.sort(dim=2).indices  # along columns (rim[row] looking at each[col] channel)
        # pick first column, where 'zero-channel' is and repeat this indices for each column
        # ig = i[..., 0].unsqueeze(3).repeat(1, 1, 1, i.size(3))
        ig = i[..., 0]
        # igm = ig[..., :self._top, :]  # pick top_k of rims least looking at 'zero-channel'
        igm = ig[..., :self._top]
        mask = torch.zeros_like(att_scores[..., 0], device=att_scores.device, dtype=torch.float32)
        # Second dim - where rims is.
        mask.scatter_(dim=2, index=igm, src=torch.ones(mask.size(), device=mask.device))
        return mask

    def rim_competition(self, att_scores: torch.Tensor):
        k_idxs = torch.topk(att_scores[..., 0], k=self._top, dim=-1, largest=False).indices
        mask = torch.zeros_like(att_scores[..., 0], device=att_scores.device, dtype=torch.float32)
        mask.scatter_(dim=2, index=k_idxs, src=torch.ones(mask.size(), device=mask.device))
        return mask.unsqueeze(3)  # from [sql, bs, rim] to [sql, bs, rim, 1] for broadcacting 1 → rim

    def stage_one(self, x: torch.Tensor, h: torch.Tensor):
        q = self.Qir(h)
        k = self.Kir(x)
        v = self.Vir(x)
        qk = torch.einsum('rsbh, bih -> sbri', q, k) / self._scale
        soft = qk.softmax(dim=3)  # rims [rows~2 dim] and inputs [cols ~3]
        with torch.no_grad():
            mask = self.rim_competition(soft)
        qkv = torch.einsum('sbri, bih -> sbrh', soft, v)
        return qkv, mask

    def stage_two(self, h: torch.Tensor, mask: torch.Tensor):
        q = self.Qrr(h)
        k = self.Krr(h)
        v = self.Vrr(h)
        # [sql, bs, rims, hid] for q, k and v - all the same.
        qk = torch.einsum('sbrh, sbch -> sbrc', q, k)  # r~c
        soft = qk.softmax(dim=3)  # att over columns
        soft = soft * mask
        qkv = torch.einsum('sbrc, sbch -> sbrh', soft, v)  # r is from q and its left intact
        return h + qkv  # residual for whatever reason

    def forward(self, x: torch.Tensor):
        collections.deque(map(lambda c: c.flatten_parameters(), self.cells), maxlen=0)
        # Add 'zero-channel' to actual channels dim
        z = torch.zeros((x.size(0), 1, x.size(2)), dtype=x.dtype, device=x.device)
        x = torch.cat((z, x), dim=1)
        # [sql, bs, chls]
        x = x.permute(2, 0, 1)
        # Stack H from all rims along new dimension: [rims, sql, bs, hid]
        h = torch.stack([c(x)[0] for c in self.cells], dim=0)
        # Get H after attentions with Inputs: [sql, bs, rims, hid]
        h_to_inp, mask = self.stage_one(x.permute(1, 2, 0), h)
        h_to_inp = NoGrad.apply(h_to_inp, mask)  # From here gradients only for top_k RIMs
        h_to_h = self.stage_two(h_to_inp, mask)
        return h_to_h * mask
