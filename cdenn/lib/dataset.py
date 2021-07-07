#!/usr/bin/env python3
import os
import mne
import copy
import numpy
import torch


class Dataset(torch.nn.Module):
    CH_to_keep = set(['EEG F4-LE', 'EEG F3-LE', 'EEG O2-LE', 'EEG Fz-LE', 'EEG C3-LE',
                      'EEG P4-LE', 'EEG P3-LE', 'EEG F8-LE', 'EEG Fp2-LE', 'EEG Pz-LE',
                      'EEG T3-LE', 'EEG C4-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG Fp1-LE',
                      'EEG F7-LE', 'EEG T6-LE', 'EEG O1-LE', 'EEG Cz-LE'])
    DROP_SUBJ = ["MDD S3 EC.edf", "MDD S30 EC.edf"]

    def __init__(self, path, standartize=True, fixed_seed=True, seed=42):
        super(Dataset, self).__init__()
        # "EO" - open eyes data and "TASK" - stimuli task data.
        file_names = [os.path.join(path, f) for f in sorted(os.listdir(path)) if "EC" in f]
        normal = [mne.io.read_raw_edf(fn, verbose=0) for fn in file_names if "H" in fn]
        anomal = [mne.io.read_raw_edf(fn, verbose=0) for fn in file_names if "MDD" in fn]
        normal = self._pre_clean(normal)
        anomal = self._pre_clean(anomal)
        # Cuts data in time by shortest subject record.
        lim = len(sorted(normal + anomal, key=lambda x: len(x))[0])
        normal = numpy.stack(([d.get_data()[:, :lim] for d in normal]), axis=0)
        anomal = numpy.stack(([d.get_data()[:, :lim] for d in anomal]), axis=0)
        self.data = torch.cat((torch.tensor(normal, dtype=torch.float32),
                               torch.tensor(anomal, dtype=torch.float32)), dim=0)
        self.labels = torch.cat((torch.zeros(size=(normal.shape[0],), dtype=torch.long),
                                 torch.ones(size=(anomal.shape[0],), dtype=torch.long)), dim=0)
        self.mixed = self.labels.clone()
        self._means = self.data.mean(dim=0, keepdim=True)
        self._std = self.data.std(dim=0, keepdim=True)
        self._device = torch.device("cpu")
        self.seed = seed
        del normal, anomal
        self._standartized = False
        self.fixed_seed = fixed_seed
        self.return_mixed = False
        self.set_sample_params(standartize=standartize)

    def to(self, device):
        self._device = torch.device(device)
        return super(Dataset, self).to(device)

    @property
    def device(self):
        return self._device

    def _switch_standartization(self):
        if self._standartized:
            self.data = self.data * self._std + self._means
            self._standartized = False
        else:
            self.data = (self.data - self._means) / self._std
            self._standartized = True

    @property
    def size(self):
        if self._window is None:
            idxs = torch.cat([t for i, t in enumerate(self.cv_idxs) if i != self.fold], dim=0)
        else:
            idxs = torch.cat([t for i, t in enumerate(self.seg_cv_idxs) if i != self.fold], dim=0)
        bs = self._bs if self._bs > 0 else idxs.size(0)
        return len(idxs.split(bs))

    def _pre_clean(self, data):
        """ Drops unnecessary channels and subjects. """
        idxs = list()
        for i, s in enumerate(data):
            s.drop_channels(list(set(s.ch_names) - self.CH_to_keep))
            if s.filenames[0].split("/")[-1] in self.DROP_SUBJ:
                idxs.append(i)
        data = [d for i, d in enumerate(data) if i not in idxs]
        return data

    def _mixup(self):
        """ Half of True labels turns to False and vice versa for each CV-fold. """
        torch.manual_seed(self.seed)
        cv_labels = [self.labels[cv] for cv in self.cv_idxs]
        for cv in cv_labels:
            num_flips = cv.size(0) // 4
            from_idxs = (cv == 1).nonzero().view(-1)
            to_idxs = (cv == 0).nonzero().view(-1)
            idxs = torch.randperm(from_idxs.size(0))[:num_flips]
            cv[from_idxs[idxs]] -= 1
            cv[to_idxs[idxs]] += 1
        self.mixed = torch.cat(cv_labels, dim=0)
        return

    def _register_folds(self, num_cv):
        """ Creates Cross Validation folds with equal amount of true/false class labels in test-fold. """
        assert 2 <= num_cv <= 10, "Set it from 2 to 10, because this range seems reasonable."
        torch.manual_seed(self.seed)
        cv_normal = torch.randperm((self.labels == 0).sum()).chunk(num_cv)
        cv_anomal = (torch.randperm((self.labels == 1).sum()) + (self.labels == 0).sum()).chunk(num_cv)
        self.cv_idxs = [torch.cat((n, a), dim=0)[torch.randperm(n.size(0) + a.size(0))] for n, a in zip(cv_normal, cv_anomal)]
        self.seg_cv_idxs = copy.deepcopy(self.cv_idxs)
        self.fold = 0
        return

    def _clean_buffers(self):
        for name in ["seg_data", "seg_mixed", "seg_labels"]:
            delattr(self, name)
        torch.cuda.empty_cache()
        return

    def set_sample_params(self, num_cv=4, bs=-1, window=None, step=None, shuffle=True, standartize=False):
        """ Optionally segments data with sliding window and forms batches. """
        self.shuffle = shuffle
        self._cv = num_cv
        self._bs = bs
        self._window = self.data.size(2) if window is None else window
        self._step = self._window if step is None else step
        self._register_folds(num_cv)
        self._mixup()
        try:
            self._clean_buffers()
        except AttributeError:
            pass
        if standartize != self._standartized:
            self._switch_standartization()
        data = self.data.unfold(dimension=2, size=self._window, step=self._step).permute(0, 2, 1, 3)
        samples_per_subj = data.size(1)
        labels = self.labels.unsqueeze(1).repeat(1, data.size(1)).view(-1)
        mixed = self.mixed.unsqueeze(1).repeat(1, data.size(1)).view(-1)
        data = data.reshape(-1, *data.size()[2:])
        idxs = torch.arange(data.size(0))
        idxs = torch.stack(idxs.split(samples_per_subj), dim=0)
        idxs = [idxs[i].view(-1) for i in self.cv_idxs]
        self.register_buffer("seg_data", data)
        self.register_buffer("seg_labels", labels)
        self.register_buffer("seg_mixed", mixed)
        self.seg_cv_idxs = idxs
        if self.device.type == "cuda":
            self.to(self.device)
        return

    def get_train(self, batched=True):
        idxs = torch.cat([t for i, t in enumerate(self.seg_cv_idxs) if i != self.fold], dim=0)
        if self.shuffle and self.fixed_seed:
            torch.manual_seed(self.seed)
            idxs = idxs[torch.randperm(idxs.size(0))]
        labels = self.seg_mixed if self.return_mixed else self.seg_labels
        bs = self._bs if self._bs > 0 else idxs.size(0)
        split = idxs.split(bs) if batched else [idxs]
        for b in split:
            yield self.seg_data[b], labels[b]

    def get_test(self, batched=False):
        idxs = self.seg_cv_idxs[self.fold]
        labels = self.seg_mixed if self.return_mixed else self.seg_labels
        bs = self._bs if self._bs > 0 else idxs.size(0)
        split = idxs.split(bs) if batched else [idxs]
        for b in split:
            yield self.seg_data[b], labels[b]
