import numpy as np
import torch
import torch.nn as nn
# import warnings
# warnings.filterwarnings("ignore")

# logpdf of independent normal distribution.
# x is of size (n_sample, n_param).
# loc and scale are int or numpy.ndarray of size n_param.
# output is of size n_sample.
def norm_logpdf(x, loc=0, scale=1):
    logpdf = (-np.log(np.sqrt(2 * np.pi) * scale) 
              - (x - loc) ** 2 / 2 / scale ** 2)
    return logpdf.sum(axis=-1)

# pdf of independent normal distribution.
def norm_pdf(x, loc=0, scale=1):
    return np.exp(norm_logpdf(x, loc, scale))

# logpdf of uniform distribution.
def uniform_logpdf(x, low=0, high=1):
    return np.log(uniform_pdf(x, low, high))

# pdf of uniform distribution.
def uniform_pdf(x, low=0, high=1):
    pdf = ((x >= low) * (x <= high)) / (high - low)
    return pdf.prod(axis=1)

# Construct neural network
class Net(nn.Module):
    def __init__(self, dimns, activate, bounds):
        super().__init__()
        layers = []
        for i in range(len(dimns) - 1):
            layers.append(nn.Linear(dimns[i], dimns[i + 1]))
            if i < len(dimns) - 2:
                layers.append(activate)
        self.net = nn.Sequential(*layers)
        self.bounds = torch.from_numpy(bounds)
        self.has_inf = torch.isinf(self.bounds).sum()

    def forward(self, x):
        x = self.net(x)
        if self.has_inf:
            x = torch.maximum(x, self.bounds[:, 0])
            x = torch.minimum(x, self.bounds[:, 1])
        else:
            x = (torch.sigmoid(x) * (self.bounds[:, 1] - 
                                     self.bounds[:, 0]) + self.bounds[:, 0])
        return x
