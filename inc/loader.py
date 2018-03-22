import numpy as np
import torch
from torch import Tensor, LongTensor, cuda
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):
    def __init__(self, d, p):
        self.d = d
        self.p = p
        # vectors normalized on the sqrt(d)-sphere
        self.sigma = 1 # => each sample data has norm ~ sqrt(d)
        self.data = self.sigma*torch.randn(p, d)
        self.labels = torch.rand(p, 1).apply_(lambda x: 1 if x > 0.5 else -1).type(LongTensor)

    def __len__(self):
        return self.p

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    
class RandomSampler:
    def __init__(self, length):
        self.length = length

    def __iter__(self):
        return iter(np.random.choice(self.length, size = self.length))

    def __len__(self):
        return self.length


def load_batch(loader, cuda = False, only_one_epoch = False):

    while True:
        for data, target in iter(loader):
            if  cuda: data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            yield data, target

        if only_one_epoch: break  # exit the loop if only_one_epoch == True
