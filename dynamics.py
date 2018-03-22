import os
import pickle
import numpy as np
from decimal import Decimal
import torch
from torch import Tensor, LongTensor, nn, optim, cuda
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from inc.models import two_layers, weights_renorm
from inc.loader import RandomDataset, RandomSampler, load_batch
from inc.compute import compute_loss, compute_grads
from inc.train import optim_step, train_hebb


print(" -- alpha = 10 -- ")


d = 128
alpha = 10
dataset = RandomDataset(d, alpha*d)
model = two_layers(d, 2*d)

history = train_hebb(model, dataset, lr = 1e-2, bs = len(dataset))

pdf = PdfPages('plots/d=128_a=10.pdf')
plt.figure(figsize = (10, 6))

plt.subplot(121)
plt.plot([ t for t, l, g_m, g_s in history ], [ l for t, l, g_m, g_s in history ], label = "Loss")
plt.xscale('log')
plt.legend()

plt.subplot(222)
plt.plot([ t for t, l, g_m, g_s in history ],
         [ g_m['hidden.weight'].norm(p = 2, dim = 1).mean() for t, l, g_m, g_s in history ],
         label = "Hidden mean")
plt.plot([ t for t, l, g_m, g_s in history ],
         [ g_s['hidden.weight'].mean() for t, l, g_m, g_s in history ], 
         label = "Hidden std")
plt.xscale('log')
plt.legend()

plt.subplot(224)
plt.plot([ t for t, l, g_m, g_s in history ],
         [ g_m['output.weight'].norm(p = 2,dim = 1).mean() for t, l, g_m, g_s in history ],
         label = "Output mean")
plt.plot([ t for t, l, g_m, g_s in history ],
         [ g_s['output.weight'].mean() for t, l, g_m, g_s in history ],
         label = "Output std")
plt.xscale('log')
plt.legend()

plt.show()
pdf.savefig()
pdf.close()


print(" -- alpha = 1 -- ")


d = 128
alpha = 1
dataset = RandomDataset(d, alpha*d)
model = two_layers(d, 2*d)

history = train_hebb(model, dataset, lr = 1e-2, bs = len(dataset))

pdf = PdfPages('plots/d=128_a=1.pdf')
plt.figure(figsize = (10, 6))

plt.subplot(121)
plt.plot([ t for t, l, g_m, g_s in history ], [ l for t, l, g_m, g_s in history ], label = "Loss")
plt.xscale('log')
plt.legend()

plt.subplot(222)
plt.plot([ t for t, l, g_m, g_s in history ],
         [ g_m['hidden.weight'].norm(p = 2, dim = 1).mean() for t, l, g_m, g_s in history ],
         label = "Hidden mean")
plt.plot([ t for t, l, g_m, g_s in history ],
         [ g_s['hidden.weight'].mean() for t, l, g_m, g_s in history ], 
         label = "Hidden std")
plt.xscale('log')
plt.legend()

plt.subplot(224)
plt.plot([ t for t, l, g_m, g_s in history ],
         [ g_m['output.weight'].norm(p = 2,dim = 1).mean() for t, l, g_m, g_s in history ],
         label = "Output mean")
plt.plot([ t for t, l, g_m, g_s in history ],
         [ g_s['output.weight'].mean() for t, l, g_m, g_s in history ],
         label = "Output std")
plt.xscale('log')
plt.legend()

plt.show()
pdf.savefig()
pdf.close()

