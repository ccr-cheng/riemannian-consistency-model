import copy
from copy import deepcopy

import torch

from training.manifolds import get_manifold


def clip_jvp(jvp: torch.Tensor, max_jvp_norm) -> torch.Tensor:
    if max_jvp_norm is None:
        return jvp
    jvp_norm = torch.linalg.vector_norm(jvp.reshape(jvp.shape[0], -1), dim=1)
    clip_coefficient = torch.clamp(max_jvp_norm / (jvp_norm + 1.e-6), max=1)
    return jvp * clip_coefficient.reshape(jvp.shape[0], *[1, ] * (len(jvp.shape) - 1))


class FlowLoss:
    def __init__(self, N, manifold, tmax=1.0):
        self.manifold = get_manifold(manifold, ndim=N)
        self.tmax = tmax

    def __call__(self, net, x):
        t = torch.rand(x.size(0), device=x.device) * self.tmax
        n = self.manifold.rand(*x.shape, device=x.device)
        xt, vf = self.manifold.vecfield(n, x, t[:, None])
        pred_vf = net(xt, t)
        diff = pred_vf - vf
        loss = self.manifold.inner(diff, diff, xt)
        return loss


class ConsistencyLoss:
    def __init__(
            self,
            N,
            manifold,
            simplified=False,
            tmax=0.995,
            tangent_warmup_steps=1,
            jvp_max_norm=10.0,
            distillation=False,
            teacher_model=None,
    ):
        if distillation:
            assert teacher_model is not None, 'Teacher model must be provided for distillation.'
        self.manifold = get_manifold(manifold, ndim=N)
        self.simplified = simplified
        self.tmax = tmax
        self.tangent_warmup_steps = tangent_warmup_steps
        self.jvp_max_norm = jvp_max_norm
        self.distillation = distillation
        self.teacher_model = teacher_model

    def __call__(self, net, x, iter_steps):
        t = torch.rand(x.size(0), device=x.device) * self.tmax
        t_expand = t[:, None, None]
        n = self.manifold.rand(*x.shape, device=x.device)
        xt, vf = self.manifold.vecfield(n, x, t[:, None])
        if self.distillation:
            with torch.no_grad():
                vf = self.teacher_model(xt, t)

        # Here, we need to modify the tangent vector with the Jacobian to account for the potential coordinate transform.
        # For SO(3), the 3-vector representation is NOT the canonical Riemannian coordinate, so there will be a Jacobian term.
        # For Torus and Sphere, the ambient coordinates are Riemannian, so no Jacobian is needed (Jacobian is identity).
        tangents = (
            self.manifold.right_jac_inv(xt, vf) if hasattr(self.manifold, 'right_jac_inv') else vf,  # dx
            torch.ones_like(t)  # dt
        )

        # EDM2 modifies the parameters inplace, which will fail the forward-mode JVP calculation.
        # If you are not using EDM2, you may consider using torch.func.jvp for potentially better efficiency.
        pred_vf, dvf = torch.autograd.functional.jvp(net, (xt, t), tangents, create_graph=True)
        # pred_vf, dvf = torch.func.jvp(net, (xt, t), tangents)
        dvf = dvf.detach()
        pred_vf_detach = pred_vf.detach()
        u = (1 - t_expand) * pred_vf
        pred_x1 = self.manifold.exp(xt, u)
        pred_x1_detach = pred_x1.detach()
        with torch.no_grad():
            # tangent warmup
            r = min(1.0, iter_steps + 1 / self.tangent_warmup_steps)
            cov_deriv = self.manifold.cov_deriv(pred_vf_detach, dvf, vf, xt).detach()
            du = -pred_vf_detach + (1 - t_expand) * cov_deriv * r
            if not self.simplified:
                dexp_x = self.manifold.dexp_x(xt, u, du)
                dexp_u = self.manifold.dexp_u(xt, u, vf)
                g = (dexp_x + dexp_u).detach()
            else:
                g = (vf + du).detach()

        # tangent normalization
        g_normed = clip_jvp(g, self.jvp_max_norm).detach()
        if not self.simplified:
            loss = self.manifold.inner(
                pred_x1_detach - pred_x1 + g_normed, g_normed, pred_x1_detach
            ) * (t / (1 - t)).unsqueeze(-1).square()
        else:
            loss = self.manifold.inner(
                pred_vf.detach() - pred_vf + g_normed, g_normed, xt
            ) * (t / (1 - t)).unsqueeze(-1).square()
        return loss


class DiscreteConsistencyLoss:
    def __init__(
            self,
            N,
            manifold,
            tmax=0.99,
            dt=0.01,
            distillation=False,
            teacher_model=None,

    ):
        if distillation:
            assert teacher_model is not None, 'Teacher model must be provided for distillation.'
        self.manifold = get_manifold(manifold, ndim=N)
        self.distillation = distillation
        self.teacher_model = teacher_model
        self.dt = dt
        self.tmax = tmax

    def __call__(self, net, x, iter_steps):
        net_clone = deepcopy(net)  # workaround for inplace modification in EDM2
        t = torch.rand(x.size(0), device=x.device) * self.tmax
        t_expand = t[:, None, None]
        n = self.manifold.rand(*x.shape, device=x.device)
        xt, vf = self.manifold.vecfield(n, x, t[:, None])
        if self.distillation:
            with torch.no_grad():
                vf = self.teacher_model(xt, t)

        pred_x1 = self.manifold.exp(xt, (1 - t_expand) * net(xt, t))
        with torch.no_grad():
            xt_hat = self.manifold.exp(xt, self.dt * vf)
            pred_x1_hat = self.manifold.exp(
                xt_hat, (1 - t_expand - self.dt) * net_clone(xt_hat, t + self.dt)
            ).detach()
            del net_clone

        loss = self.manifold.norm2(
            self.manifold.log(pred_x1, pred_x1_hat), pred_x1.detach()
        ) / self.dt * (t / (1 - t)).unsqueeze(-1)
        return loss
