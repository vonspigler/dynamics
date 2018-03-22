import numpy as np
from decimal import Decimal
import torch
from torch import Tensor, LongTensor, nn
from torch.nn import functional as F


class two_layers(torch.nn.Module):
    def __init__(self, d, hidden, bias = 0, fixed_bias = True):
        # IMPLEMENT bias
        super().__init__()
        self.bias = bias
        self.d = d
        self.hidden = nn.Linear(d, hidden, False)
        self.output = nn.Linear(hidden, 1, False)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


def weights_renorm(d):
    def renorm(module):
        classname = module.__class__.__name__

        if classname.find('Linear') != -1:
            size = module.out_features
            if size == 1:
                module.weight.data.div_(module.weight.data.norm()/np.sqrt(d))
            else:
                module.weight.data = module.weight.data \
                    / module.weight.data.norm(p = 2, dim = 1).expand(module.in_features, module.out_features).t() \
                    * np.sqrt(d)
    return renorm
