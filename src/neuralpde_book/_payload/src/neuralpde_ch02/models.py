"""Book 2.2.2 and 2.3.2: y_theta(x) = G(x) + D(x) N_theta(2x-1).

G lifts the Dirichlet data g; D vanishes on the boundary. The input rescaling
belongs to the network, while derivatives are taken in physical coordinates x.
Adapted from ocpde_dl/pinn/func_net.py; architecture and initialization retained.
"""
import torch
from torch import nn


def gradient(y, x):
    """Spatial gradient of a pointwise scalar field: (N, 1) -> (N, dim).

    The batch samples must be independent, as in CoordinateMLP. This computes
    derivatives with respect to x, not parameter gradients with respect to theta.
    """
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def laplacian(y, x):
    """Return Delta y with shape (N, 1); retain the graph for training."""
    grad_y = gradient(y, x)
    return sum(torch.autograd.grad(grad_y[:, i:i+1], x, torch.ones_like(grad_y[:, i:i+1]),
                                  create_graph=True, retain_graph=True)[0][:, i:i+1]
               for i in range(x.shape[1]))


class CoordinateMLP(nn.Module):
    """Boundary-compatible solution model; ``net`` is the unconstrained MLP.

    The existing ``net.*`` state-dict keys are retained so frozen model weights
    can be loaded without conversion. ``model(x)`` returns y_theta, not N_theta.
    """

    def __init__(self, problem, width, depth):
        super().__init__()
        self.problem = problem
        layers = []
        input_width = problem.dim
        for _ in range(depth):
            layers.extend((nn.Linear(input_width, width), nn.Tanh()))
            input_width = width
        layers.append(nn.Linear(input_width, 1))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Physical coordinates x -> neural solution y_theta, shape (N, 1)."""
        return self.problem.lifting(x) + self.problem.boundary_factor(x)*self.net(2*x-1)
