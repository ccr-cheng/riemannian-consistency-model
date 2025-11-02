import torch
from tqdm import tqdm

from training.manifolds import Manifold


def flow_sample(
        model,
        manifold: Manifold,
        n_sample,
        data_dim=1,
        n_steps=1000,
        device=None,
        return_traj=False,
):
    """
    Euler solver for Riemannian flow matching sampling.
    :param model: Riemannian flow matching model.
    :param manifold: instance of Manifold.
    :param n_sample: number of samples to generate.
    :param data_dim: data dimension.
    :param n_steps: number of Euler steps.
    :param device: device to use.
    :param return_traj: if True, return the whole trajectory.
    :return: generated samples of size (B, D, n) if not return_traj, else (N+1, B, D, n).
    """
    x0 = manifold.rand(n_sample, data_dim, manifold.ndim, device=device)
    model = model.to(device)
    xs = [x0]
    x = x0
    ts = torch.linspace(0, 1, n_steps + 1, device=device)
    dt = 1 / n_steps
    for t in tqdm(ts[:-1]):
        pred_vf = model(x, torch.full((n_sample,), t, device=device))
        x = manifold.exp(x, pred_vf * dt)
        if return_traj:
            xs.append(x)
    return torch.stack(xs, dim=0) if return_traj else x


def consistency_sample(
        model,
        manifold: Manifold,
        n_sample,
        data_dim=1,
        n_steps=2,
        mid_t=0.8,
        device=None,
        resample_noise=True,
        return_traj=False,
):
    """
    Consistency model sampling.
    :param model: Riemannian consistency model.
    :param manifold: instance of Manifold.
    :param n_sample: number of samples to generate.
    :param data_dim: data dimension.
    :param n_steps: number of steps.
    :param mid_t: optional intermediate time t. If provided, the first step always goes from t=0 to t=mid_t.
                  If not provided, uniform steps from t=0 to t=1 are used. Default is 0.8.
    :param device: device to use.
    :param resample_noise: if True, resample noise at each step.
    :param return_traj: if True, return the whole trajectory.
    :return: generated samples of size (B, D, n) if not return_traj, else (N+1, B, D, n).
    """
    x0 = manifold.rand(n_sample, data_dim, manifold.ndim, device=device)
    model = model.to(device)
    xs = [x0]
    x = x0

    if mid_t is not None:
        ts = torch.cat([
            torch.tensor([0.], device=device),
            torch.linspace(mid_t, 1.0, n_steps, device=device)
        ], dim=0)
    else:
        ts = torch.linspace(0, 1, n_steps + 1, device=device)
    for i, (t_cur, t_next) in enumerate(zip(ts[:-1], ts[1:])):
        x = manifold.exp(x, (1 - t_cur) * model(x, torch.full((n_sample,), t_cur, device=device)))
        if t_next < 1:
            new_x0 = manifold.rand(n_sample, data_dim, manifold.ndim, device=device) if resample_noise else x0
            x = manifold.interpolate(new_x0, x, torch.full((n_sample, 1), t_next, device=device))
        if return_traj:
            xs.append(x)
    return torch.stack(xs, dim=0) if return_traj else x
