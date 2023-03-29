import torch
import torch.nn as nn

class InstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(InstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))

    def forward(self, x):
        # Calculate mean and variance across height and width dimensions
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.var(x, dim=(2, 3), keepdim=True)

        # Normalize the input x
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift the normalized input
        out = self.weight * x_norm + self.bias
        return out
