import numpy as np
import torch
from torch import Tensor, LongTensor, cuda#, nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import collections

from .loader import RandomDataset, RandomSampler, load_batch


def compute_loss(model, dataset, bs):
    loader = DataLoader(
        dataset,
        batch_size = bs,
        pin_memory = cuda.is_available(),
        sampler = RandomSampler(len(dataset))
    )

    tot_loss = 0
    for input, target in load_batch(loader, cuda = cuda.is_available(), only_one_epoch = True):
        output = model(input)
        loss = -(output*target.type(Tensor)).sum()  # hebb's rule
        tot_loss += loss.data[0]

    return tot_loss / len(dataset)


def compute_grads(model, dataset, bs_grad, bs):
    for p in model.parameters(): p.grad = None

    loader = DataLoader(
        dataset,
        batch_size = bs,
        pin_memory = cuda.is_available(),
        sampler = RandomSampler(len(dataset))
    )
    # compute gradient
    for input, target in load_batch(loader, cuda = cuda.is_available(), only_one_epoch = True):
        output = model(input)
        loss = -(output*target.type(Tensor)).mean()  # hebb's rule
        loss.backward()

    state = { key: value for key, value in model.state_dict(keep_vars = True).items()
              if isinstance(value, Variable) and value.requires_grad }
    grad_mean = collections.OrderedDict(
        [ (key, value.grad.data.clone()) for key, value in state.items() ]
    )
    grad_std = collections.OrderedDict(
        [ (key, torch.zeros_like(value.data)) for key, value in state.items() ]
    )

    n = 0
    loader = DataLoader(
        dataset,
        batch_size = 1,
        pin_memory = cuda.is_available(),
        sampler = RandomSampler(len(dataset))
    )
    # compute variance
    for input, target in load_batch(loader, cuda = cuda.is_available(), only_one_epoch = True):
        output = model(input)

        for p in model.parameters(): p.grad = None
        loss = -(output*target.type(Tensor)).mean()  # hebb's rule
        loss.backward()

        gradient = collections.OrderedDict(
            [ (key, value.grad.data) for key, value in state.items() ]
        )

        for key in state:
#            print(gradient[key])
            grad_std[key] += (gradient[key] - grad_mean[key])**2

        n += 1
        if n >= bs_grad and n > 0:
            break

    for key in state:
        grad_std[key] = (grad_std[key] / n)**0.5

    return grad_mean, grad_std
