"""Instance-wise ports. Smooth state ansatz replaces legacy CNP straight-through clamp.

Book/code notation:
Book (5.7)-(5.10), (5.22): q_scaled=p/alpha, lambda_admm is the unscaled multiplier for u-z, and z is the feasible split control. The q_net and lam storage keys stay unchanged for existing checkpoints.
"""

import copy
import math
import torch
from torch import nn
from .problem import clip


def mlp(width=64, depth=6):
    layers = [nn.Linear(2, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    return nn.Sequential(*layers, nn.Linear(width, 1))


def boundary_factor(x):
    return 16 * x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])


def laplace(v, x):
    grad_value = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    return sum(
        torch.autograd.grad(
            grad_value[:, i : i + 1],
            x,
            torch.ones_like(grad_value[:, i : i + 1]),
            create_graph=True,
        )[0][:, i : i + 1]
        for i in range(2)
    )


class Fields(nn.Module):
    def __init__(self, problem, method, width=64, depth=6):
        super().__init__()
        self.problem, self.method = problem, method
        self.y_net = mlp(width, depth)
        if method != "cnp":
            self.q_net = mlp(width, depth)

    def y(self, x):
        boundary_weight = boundary_factor(x)
        if self.method == "cnp":
            return (
                self.problem.state_cap
                - (math.sqrt(self.problem.state_cap) + boundary_weight * self.y_net(x)) ** 2
            )
        return boundary_weight * self.y_net(x)

    def q(self, x):
        return boundary_factor(x) * self.q_net(x)

    def u(self, x):
        if self.method == "cnp":
            return -laplace(self.y(x), x)  # f=0
        return clip(-self.q(x), self.problem.lower, self.problem.upper)


class SplitState:
    """Pointwise ADMM history. Saved q networks recover z/lambda on ANY coordinates.

    Uses book's +<lambda,u-z> convention. Each snapshot is the model AFTER its
    inner solve and BEFORE the corresponding outer update. No OSNN substitution.
    """

    def __init__(self, problem, beta):
        self.problem, self.beta, self.snapshots = problem, beta, []

    def append(self, model):
        frozen = copy.deepcopy(model.q_net).eval()
        for parameter in frozen.parameters():
            parameter.requires_grad_(False)
        self.snapshots.append(frozen)

    def at(self, x):
        z = torch.full_like(x[:, 0:1], min(max(0, self.problem.lower), self.problem.upper))
        lambda_admm = torch.zeros_like(z)
        u, previous_z = z, z
        for qnet in self.snapshots:
            previous_z = z
            q_scaled = boundary_factor(x) * qnet(x)
            u = (self.beta * z - lambda_admm - self.problem.alpha * q_scaled) / (
                self.problem.alpha + self.beta
            )
            z = clip(u + lambda_admm / self.beta, self.problem.lower, self.problem.upper)
            lambda_admm = lambda_admm + self.beta * (u - z)
        return dict(u=u, z=z, lam=lambda_admm, primal=u - z, dual=self.beta * (z - previous_z))

    def state_dict(self):
        return [n.state_dict() for n in self.snapshots]

    def load(self, states, model):
        for state in states:
            self.append(model)
            self.snapshots[-1].load_state_dict(state)


def loss_terms(model, x, split_values=None, beta=None):
    problem = model.problem
    y = model.y(x)
    if model.method == "cnp":
        u = -laplace(y, x)
        return dict(
            objective=0.5 * ((y - problem.sine(x)) ** 2).mean()
            + 0.5 * problem.alpha * (u**2).mean()
        )
    q_scaled = model.q(x)
    if split_values is None:
        u = clip(-q_scaled, problem.lower, problem.upper)
    else:
        z, lambda_admm = split_values
        u = (beta * z - lambda_admm - problem.alpha * q_scaled) / (problem.alpha + beta)
    state_residual = -laplace(y, x) - u - problem.f(x)
    adjoint_residual = -problem.alpha * laplace(q_scaled, x) - y + problem.yd(x)
    return dict(
        state=state_residual.square().mean(),
        adjoint=adjoint_residual.square().mean() / problem.alpha,
    )
