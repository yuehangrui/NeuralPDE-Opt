"""Coordinate MLP, hard constraints and autograd, adapted from ocpde_dl/pinn/func_net.py."""
import torch
from torch import nn


def gradient(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def laplacian(y, x):
    g = gradient(y, x)
    return sum(torch.autograd.grad(g[:, i:i+1], x, torch.ones_like(g[:, i:i+1]),
                                  create_graph=True, retain_graph=True)[0][:, i:i+1]
               for i in range(x.shape[1]))


class CoordinateMLP(nn.Module):
    def __init__(self, problem, width, depth):
        super().__init__()
        self.problem = problem
        layers = []
        last = problem.dim
        for _ in range(depth):
            layers.extend((nn.Linear(last, width), nn.Tanh()))
            last = width
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.problem.lifting(x) + self.problem.boundary_factor(x)*self.net(2*x-1)
