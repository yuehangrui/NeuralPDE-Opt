"""Legacy Tanh trial spaces; physical p is always exposed as p=alpha*q.

Book/code notation:
Book 4.4.1: Fields returns (y,u,p) in physical units. q_scaled=p/alpha is internal; control is -q_scaled. State residual is -Delta y-u-f and adjoint residual is -Delta p-y+yd.
"""

import math
import torch
from torch import nn


def mlp(width, depth):
    if width < 1 or depth < 1:
        raise ValueError("width and depth must be positive")
    layers = [nn.Linear(2, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    return nn.Sequential(*layers, nn.Linear(width, 1))


def boundary_factor(x):
    return 16 * x[:, :1] * (1 - x[:, :1]) * x[:, 1:2] * (1 - x[:, 1:2])


def laplace(value, x):
    grad_value = torch.autograd.grad(value, x, torch.ones_like(value), create_graph=True)[0]
    result = 0
    for j in range(2):
        result = (
            result
            + torch.autograd.grad(
                grad_value[:, j : j + 1],
                x,
                torch.ones_like(grad_value[:, j : j + 1]),
                create_graph=True,
            )[0][:, j : j + 1]
        )
    return result


class Fields(nn.Module):
    def __init__(self, method, alpha=1e-3, width=64, depth=6):
        super().__init__()
        if method not in ("direct", "kkt", "alm"):
            raise ValueError("unknown method")
        self.method, self.alpha = method, float(alpha)
        self.y_net, self.second_net = mlp(width, depth), mlp(width, depth)

    def forward(self, x):
        y = boundary_factor(x) * self.y_net(x)
        second = self.second_net(x)
        if self.method == "kkt":
            q_scaled = boundary_factor(x) * second  # legacy q=p/alpha, u=-q
            return y, -q_scaled, self.alpha * q_scaled
        return y, second, None


def residuals(model, problem, x):
    y, u, p = model(x)
    state_residual = -laplace(y, x) - u - problem.f(x)
    adjoint_residual = None if p is None else -laplace(p, x) - y + problem.yd(x)
    return state_residual, adjoint_residual


def loss_terms(model, problem, x, penalty=1.0, multiplier=None):
    y, u, p = model(x)
    state_residual = -laplace(y, x) - u - problem.f(x)
    objective = 0.5 * (y - problem.yd(x)).square().mean() + 0.5 * problem.alpha * u.square().mean()
    if model.method == "kkt":
        adjoint_residual = -laplace(p, x) - y + problem.yd(x)
        # Identical scaling to legacy (-alpha*Delta(q)-y+yd)/sqrt(alpha).
        loss = state_residual.square().mean() + adjoint_residual.square().mean() / problem.alpha
    elif model.method == "alm":
        if multiplier is None or multiplier.shape != state_residual.shape:
            raise ValueError("one ALM multiplier is required for each fixed training point")
        loss = (
            objective
            + penalty * state_residual.square().mean()
            + (multiplier * state_residual).mean()
        )
    else:
        loss = objective + penalty * state_residual.square().mean()
    return loss, objective, state_residual


def multiplier_update(multiplier, residual, old_mu):
    if not math.isfinite(old_mu) or old_mu <= 0 or multiplier.shape != residual.shape:
        raise ValueError("invalid ALM update")
    return (multiplier + 2 * old_mu * residual.detach()).detach()
