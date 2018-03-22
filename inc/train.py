import numpy as np
import torch
from torch import Tensor, LongTensor, cuda, optim#, nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import collections

from .loader import RandomDataset, RandomSampler, load_batch
from .compute import compute_loss, compute_grads
from .models import weights_renorm


def optim_step(model, optimizer, loss, weights_renorm):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.apply(weights_renorm(model.d))


def train_hebb(model, dataset, lr, bs):
    optimizer = optim.SGD(model.parameters(), lr = lr)

    history = []
    model.apply(weights_renorm(model.d))
    if cuda.is_available(): model.cuda()    
    loader = DataLoader(
        dataset,
        batch_size = bs,
        pin_memory = cuda.is_available(),
        sampler = RandomSampler(len(dataset))
    )

    next_time = 1
    time = 0
    for input, target in load_batch(loader, cuda = cuda.is_available()):
        time += 1
        if time > 10000: break

        output = model(input)
        loss = -(output*target.type(Tensor)).mean()  # hebb's rule
        optim_step(model, optimizer, loss, weights_renorm)

        if time > next_time:
            avg_loss = compute_loss(model, dataset, bs = len(dataset))
            grad_avg, grad_std = compute_grads(model, dataset, -1, bs = len(dataset))
            history.append([lr*time, avg_loss, grad_avg, grad_std])
            next_time *= 1.2

    return history
