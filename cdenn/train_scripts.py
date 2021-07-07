#!/usr/bin/env python3
import tqdm
import copy
import torch
from cdenn.lib import *


DEFAULT_SEED = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def trainable_params(net: torch.nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    return sum([torch.tensor(p.size()).prod() for p in model_parameters])


def pr(x, y):
    """ Metrics calculation from: https://en.wikipedia.org/wiki/Confusion_matrix
        Returns precision, recall, specificity and f1 (in that order)
    """
    tp = ((x == y) * (x == 1)).sum().to(torch.float32)
    tn = ((x == y) * (x == 0)).sum().to(torch.float32)
    fp = ((x != y) * (x == 1)).sum().to(torch.float32)
    fn = ((x != y) * (x == 0)).sum().to(torch.float32)
    pr = tp / (tp + fp)
    rc = tp / (tp + fn)
    sp = tn / (tn + fp)
    f1 = (2 * tp) / (2 * tp + fp + fn)
    return pr, rc, sp, f1


def test_cv(net: torch.nn.Module, ds: Dataset, cv: int, batched=False, pt=True):
    loss_fct = torch.nn.CrossEntropyLoss()
    net.eval()
    # BatchNorm does not work in eval mode
    # More info on subject: https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/32
    for p in net.modules():
        if type(p) in [torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm]:
            p.train(True)
        # PreNorm in training mode is INCORRECT, but it was done in report
        if (type(p) is PreNorm) and pt:
            p.train(True)
    with torch.no_grad():
        if batched:
            ys, ps = list(), list()
            for x, y in ds.get_train():
                p = net(x)
                ps.append(p.detach().cpu())
                ys.append(y.detach().cpu())
            p = torch.cat(ps, dim=0)
            y = torch.cat(ys, dim=0)
        else:
            x, y = next(ds.get_train(batched=False))
            p = net(x)
        l_tr = loss_fct(p, y)
        pre, rec, spe, f1_tr = pr(p.softmax(-1).argmax(-1), y)
        print(f"Train[{cv}]")
        print(f"Loss: {l_tr.item():.2f}|Precision→ {pre:.2f}|Recall→ {rec:.2f}|Specificity→ {spe:.2f}|F1→ {f1_tr:.2f}")
        # Test should fit in memory anyway, there is no reason for batched version
        x, y = next(ds.get_test())
        p = net(x)
        l_te = loss_fct(p, y)
        pre, rec, spe, f1_te = pr(p.softmax(-1).argmax(-1), y)
        print(f"Test[{cv}]")
        print(f"Loss: {l_te.item():.2f}|Precision→ {pre:.2f}|Recall→ {rec:.2f}|Specificity→ {spe:.2f}|F1→ {f1_te:.2f}")
    return (l_tr, f1_tr), (l_te, f1_te)


def train_cv(cv: int, epochs: int, net: torch.nn.Module, opt: torch.optim.Optimizer, ds: Dataset,
             batched_test=False, lr_shedule=False, pt=True):
    bar = tqdm.tqdm(range(epochs * ds.size), desc=f"Train[{cv}]")
    loss_fct = torch.nn.CrossEntropyLoss()
    step_lr = opt.param_groups[0]["lr"] / epochs
    for _ in range(epochs):
        for x, y in ds.get_train():
            p = net(x)
            loss = loss_fct(p, y)
            loss.backward()
            opt.step()
            opt.zero_grad()
            bar.update()
            bar.set_postfix(Loss=f"{loss.item():.4f}")
        if lr_shedule:
            opt.param_groups[0]["lr"] -= step_lr
    bar.clear()
    bar.close()
    (l_tr, f1_tr), (l_te, f1_te) = test_cv(net=net, ds=ds, cv=cv, batched=batched_test, pt=pt)
    return net, (l_tr, f1_tr), (l_te, f1_te)


def bendr_train(dataset: Dataset, epochs: int, batch_size: int, num_cv: int, seed: int, lr_pretrain=5e-5,
                fixed_seed=True, lr_rest=1e-3, weight_decay=0, mixed=False, standartize=True, device="cuda:0"):
    train_stats, test_stats = list(), list()
    for i in range(num_cv):
        torch.manual_seed(DEFAULT_SEED)
        net = Bendr()
        net.to(device)
        cont_par, rest_par = list(), list()
        for n, p in net.named_parameters():
            if "contextualizer" in n:
                cont_par.append(p)
            else:
                rest_par.append(p)
        opt = torch.optim.AdamW([{"params": cont_par, "lr": lr_pretrain}, {"params": rest_par}],
                                lr=lr_rest, weight_decay=weight_decay)
        n, tr, te = train_cv(cv=i, epochs=epochs, net=net, opt=opt, ds=dataset, batched_test=True)
        train_stats.append(tr)
        test_stats.append(te)
        dataset.fold += 1
        # BENDR is too large to store models and optimizers for all CV-folds.
        del net, opt
        torch.cuda.empty_cache()
    l_tr, f1_tr = map(lambda x: torch.tensor(x), zip(*train_stats))
    l_te, f1_te = map(lambda x: torch.tensor(x), zip(*test_stats))
    print(f"All CVs Train:\nLoss: {l_tr.mean():.2f}±{l_tr.std():.2f} F1: {f1_tr.mean():.2f}±{f1_tr.std():.2f}")
    print(f"All CVs Test:\nLoss: {l_te.mean():.2f}±{l_te.std():.2f} F1: {f1_te.mean():.2f}±{f1_te.std():.2f}")
    return


def main_train(net: torch.nn.Module, dataset: Dataset, epochs: int, batch_size: int, num_cv: int, seed: int,
               window: int, device="cuda:0", standartize=False, mixed=False, lr=1e-2, wd=1e-2, lr_shedule=False,
               batched_test=False, fixed_seed=True, pre_train=True):
    nets = [copy.deepcopy(net) for _ in range(num_cv)]
    opts = [torch.optim.AdamW(params=n.parameters(), lr=lr, weight_decay=wd) for n in nets]
    train_stats, test_stats = list(), list()
    for i in range(num_cv):
        n, tr, te = train_cv(cv=i, epochs=epochs, net=nets[i], opt=opts[i], pt=pre_train,
                             ds=dataset, batched_test=batched_test, lr_shedule=lr_shedule)
        train_stats.append(tr)
        test_stats.append(te)
        dataset.fold += 1
    l_tr, f1_tr = map(lambda x: torch.tensor(x), zip(*train_stats))
    l_te, f1_te = map(lambda x: torch.tensor(x), zip(*test_stats))
    print(f"All CVs Train:\nLoss: {l_tr.mean():.2f}±{l_tr.std():.2f} F1: {f1_tr.mean():.2f}±{f1_tr.std():.2f}")
    print(f"All CVs Test:\nLoss: {l_te.mean():.2f}±{l_te.std():.2f} F1: {f1_te.mean():.2f}±{f1_te.std():.2f}")
    return
