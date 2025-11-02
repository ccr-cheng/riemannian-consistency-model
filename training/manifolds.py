from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

_EPS = 1e-4

_MANIFOLD_DICT = {}


def register_manifold(name):
    def decorator(cls):
        _MANIFOLD_DICT[name] = cls
        return cls

    return decorator


def get_manifold(m_type, **kwargs):
    m_type = m_type.lower()
    if m_type not in _MANIFOLD_DICT:
        raise ValueError(f"Manifold type '{m_type}' is not registered. Available types: {list(_MANIFOLD_DICT.keys())}")
    return _MANIFOLD_DICT[m_type](**kwargs)


class Manifold(ABC):
    """
    Base class for Riemannian manifolds. All geometric operations are class methods and dimension-agnostic,
    so one can directly access these methods on inherited classes without initialization:

    >>> x, y = Sphere.proj_x(torch.randn(2, 3)), Sphere.proj_x(torch.randn(2, 3))
    >>> u = Sphere.proj_u(torch.randn(2, 3), x)
    >>> Sphere.exp(x, u)
    >>> Sphere.log(x, y)


    :param ndim: dimension of the manifold
    :param kwargs: additional parameters
    """

    def __init__(self, ndim, **kwargs):
        self.ndim = ndim
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f'{self.__class__.__name__}(ndim={self.ndim})'

    ###########################################################
    # Manifold Operators used in Riemannian Flow Matching     #
    ###########################################################

    @classmethod
    @abstractmethod
    def exp(cls, x: Tensor, u: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the exponential map :math:`\exp_x(u)`.
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param u: tangent vector in :math:`T_x M`, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: exponential map, Tensor of shape (..., n)
        """

    @classmethod
    @abstractmethod
    def log(cls, x: Tensor, y: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the logarithm map :math:`\log_x(y)`.
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param y: target point on the manifold, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: logarithm map, Tensor of shape (..., n)
        """

    @classmethod
    @abstractmethod
    def proj_x(cls, x: Tensor, eps=_EPS) -> Tensor:
        """
        Project a point x onto the manifold.
        :param x: point in the ambient space, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: projected point on the manifold, Tensor of shape (..., n)
        """

    @classmethod
    @abstractmethod
    def proj_u(cls, u: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        """
        Project a tangent vector u onto the tangent space of x.
        :param u: tangent vector, Tensor of shape (..., n)
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: projected vector field, Tensor of shape (..., n)
        """

    @classmethod
    @abstractmethod
    def inner(cls, u: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the Riemannian metric (inner product) at point x: :math:`\langle u, v\rangle_g=g_x(u, v)`.
        :param u: tangent vector in :math:`T_x M`, Tensor of shape (..., n)
        :param v: tangent vector in :math:`T_x M`, Tensor of shape (..., n)
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: inner product, Tensor of shape (...)
        """

    @classmethod
    def norm2(cls, u: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the squared Riemannian norm of a tangent vector u at point x: :math:`\|u\|_g^2=g_x(u, u)`.
        This method might be useful for numerical stability.
        :param u: tangent vector in :math:`T_x M`, Tensor of shape (..., n)
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: squared Riemannian norm, Tensor of shape (...)
        """
        return cls.inner(u, u, x, eps)

    @classmethod
    def norm(cls, u: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the Riemannian norm of a tangent vector u at point x: :math:`\|u\|_g=\sqrt{g_x(u, u)}`.
        :param u: tangent vector in :math:`T_x M`, Tensor of shape (..., n)
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: Riemannian norm, Tensor of shape (...)
        """
        return torch.sqrt(cls.norm2(u, x, eps))

    @classmethod
    def dist(cls, x: Tensor, y: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the Riemannian distance between x and y: :math:`d_g(x, y)=\|\log_x y\|_g`.
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param y: target point on the manifold, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: Riemannian distance, Tensor of shape (...)
        """
        return cls.norm(cls.log(x, y, eps), x)

    @classmethod
    def interpolate(cls, x: Tensor, y: Tensor, t: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the geodesic interpolant between x and y at timestep t: :math:`\gamma(t)=\exp_x(t\log_x y)`.
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param y: target point on the manifold, Tensor of shape (..., n)
        :param t: timestep between 0 and 1, Tensor of shape (...)
        :param eps: small float to avoid instability
        :return: geodesic interpolant at timestep t, Tensor of shape (..., n)
        """
        return cls.exp(x, t.unsqueeze(-1) * cls.log(x, y, eps), eps)

    @classmethod
    def vecfield(cls, x: Tensor, y: Tensor, t: Tensor, eps=_EPS) -> tuple[Tensor, Tensor]:
        r"""
        Calculate the geodesic interpolant :math:`\gamma(t)` between x and y at timestep t
        and the vector field :math:`\dot\gamma(t)`.
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param y: target point on the manifold, Tensor of shape (..., n)
        :param t: timestep between 0 and 1, Tensor of shape (...)
        :param eps: small float to avoid instability
        :return:
            xt: geodesic interpolant at timestep t, Tensors of shape (..., n)
            vf: vector field, Tensors of shape (..., n)
        """
        xt = cls.interpolate(x, y, t, eps)
        vf = cls.log(xt, y, eps) / (1 - t).unsqueeze(-1)
        return xt, vf

    @classmethod
    @abstractmethod
    def rand(cls, *sizes, device=None, eps=_EPS) -> Tensor:
        """
        Generate random points on the manifold.
        :param sizes: shape of the Tensor to be returned
        :param device: device to put the Tensor on
        :param eps: small float to avoid instability
        :return: random points on the manifold, Tensor of shape sizes
        """

    ###########################################################
    # Manifold Operators used in Riemannian Consistency Model #
    ###########################################################

    @classmethod
    def cov_deriv(cls, u: Tensor, du: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the covariant derivative (affine connection) of a vector field u along v at point x.
        .. math::
            \nabla_{v}u = \dot u^k + \Gamma^k_{ij} v^i u^j

        :param u: vector field to be differentiated, Tensor of shape (..., n)
        :param du: time derivative of u, Tensor of shape (..., n)
        :param v: vector field that defines the geodesic, Tensor of shape (..., n)
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: covariant derivative, Tensor of shape (..., n)
        """
        raise NotImplementedError

    @classmethod
    def dexp_x(cls, x: Tensor, u: Tensor, du: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the differential of the exponential map at point x, tangent vector u, applied to v.
        .. math::
            d(\exp_x)_u(v)
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param u: tangent vector in :math:`T_x M`, Tensor of shape (..., n)
        :param du: target tangent vector, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: differential of the exponential map of v, Tensor of shape (..., n)
        """
        raise NotImplementedError

    @classmethod
    def dexp_u(cls, x: Tensor, u: Tensor, dx: Tensor, eps=_EPS) -> Tensor:
        r"""
        Calculate the differential of the exponential map at tangent vector u, point x, applied to v.
        .. math::
            d(\exp u)_x(v)
        :param x: base point on the manifold, Tensor of shape (..., n)
        :param u: tangent vector in :math:`T_x M`, Tensor of shape (..., n)
        :param dx: target tangent vector, Tensor of shape (..., n)
        :param eps: small float to avoid instability
        :return: differential of the exponential map of v, Tensor of shape (..., n)
        """
        raise NotImplementedError


@register_manifold('euclidean')
class Euclidean(Manifold):
    r"""Euclidean manifold :math:`\mathbb{R}^n`."""

    @classmethod
    def exp(cls, x: Tensor, u: Tensor, eps=_EPS) -> Tensor:
        return x + u

    @classmethod
    def log(cls, x: Tensor, y: Tensor, eps=_EPS) -> Tensor:
        return y - x

    @classmethod
    def proj_x(cls, x: Tensor, eps=_EPS) -> Tensor:
        return x

    @classmethod
    def proj_u(cls, u: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        return u

    @classmethod
    def inner(cls, u: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        return (u * v).sum(-1)

    @classmethod
    def vecfield(cls, x: Tensor, y: Tensor, t: Tensor, eps=_EPS) -> tuple[Tensor, Tensor]:
        return cls.interpolate(x, y, t), y - x

    @classmethod
    def rand(cls, *sizes, device=None, eps=_EPS) -> Tensor:
        return torch.randn(*sizes, device=device)

    @classmethod
    def cov_deriv(cls, u: Tensor, du: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        return du

    @classmethod
    def dexp_x(cls, x: Tensor, u: Tensor, du: Tensor, eps=_EPS) -> Tensor:
        return du

    @classmethod
    def dexp_u(cls, x: Tensor, u: Tensor, dx: Tensor, eps=_EPS) -> Tensor:
        return dx


@register_manifold('torus')
class Torus(Manifold):
    r"""N-dimensional Torus manifold :math:`T^N = (S^1)^N`,
    parameterized by angles :math:`(\theta_1, \theta_2, \dots, \theta_N)`.
    """

    def __init__(self, ndim, **kwargs):
        assert ndim >= 1, "Torus manifold must have at least 1 dimension"
        super().__init__(ndim, **kwargs)
        self.ndim = ndim

    @classmethod
    def _wrap_angle(cls, theta: Tensor) -> Tensor:
        """Wrap angles to [0, 2π)."""
        return theta % (2 * torch.pi)

    @classmethod
    def _shortest_angle(cls, theta1: Tensor, theta2: Tensor) -> Tensor:
        """Compute the shortest angular difference, accounting for periodicity in each dimension."""
        diff = theta2 - theta1
        return torch.atan2(torch.sin(diff), torch.cos(diff))

    @classmethod
    def exp(cls, x: Tensor, u: Tensor, eps=_EPS) -> Tensor:
        return cls._wrap_angle(x + u)

    @classmethod
    def log(cls, x: Tensor, y: Tensor, eps=_EPS) -> Tensor:
        return cls._shortest_angle(x, y)

    @classmethod
    def proj_x(cls, x: Tensor, eps=_EPS) -> Tensor:
        return cls._wrap_angle(x)

    @classmethod
    def proj_u(cls, u: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        return u

    @classmethod
    def inner(cls, u: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        return (u * v).sum(-1)

    @classmethod
    def dist(cls, x: Tensor, y: Tensor, eps=_EPS) -> Tensor:
        dtheta = cls._shortest_angle(x, y)
        return torch.norm(dtheta, p=2, dim=-1)

    @classmethod
    def vecfield(cls, x: Tensor, y: Tensor, t: Tensor, eps=_EPS) -> tuple[Tensor, Tensor]:
        xt = cls.interpolate(x, y, t)
        vf = cls._shortest_angle(x, y)  # Constant velocity
        return xt, vf

    @classmethod
    def rand(cls, *sizes, device=None, eps=_EPS) -> Tensor:
        r"""
        Generate random points on the torus in [0, 2π).
        """
        return 2 * torch.pi * torch.rand(sizes, device=device)

    @classmethod
    def cov_deriv(cls, u: Tensor, du: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        return du

    @classmethod
    def dexp_x(cls, x: Tensor, u: Tensor, du: Tensor, eps=_EPS) -> Tensor:
        return du

    @classmethod
    def dexp_u(cls, x: Tensor, u: Tensor, dx: Tensor, eps=_EPS) -> Tensor:
        return dx


@register_manifold('sphere')
class Sphere(Manifold):
    r"""Sphere manifold :math:`\mathbb{S}^n`."""

    @classmethod
    def exp(cls, x: Tensor, u: Tensor, eps=_EPS) -> Tensor:
        u_norm = u.norm(dim=-1, keepdim=True)
        return torch.cos(u_norm) * x + torch.sinc(u_norm / torch.pi) * u

    @classmethod
    def log(cls, x: Tensor, y: Tensor, eps=_EPS) -> Tensor:
        u = cls.proj_u(y - x, x, eps)
        u_norm = u.norm(dim=-1, keepdim=True).clamp(min=eps)
        dist = cls.dist(x, y, eps).unsqueeze(-1).detach()
        return torch.where(dist > eps, u * dist / u_norm, u)

    @classmethod
    def proj_x(cls, x: Tensor, eps=_EPS) -> Tensor:
        return F.normalize(x, p=2, dim=-1)

    @classmethod
    def proj_u(cls, u: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        return u - (x * u).sum(dim=-1, keepdim=True) * x

    @classmethod
    def inner(cls, u: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        return (u * v).sum(-1)

    @classmethod
    def dist(cls, x: Tensor, y: Tensor, eps=_EPS) -> Tensor:
        return torch.acos((x * y).sum(-1).clamp(-1 + eps, 1 - eps))

    @classmethod
    def vecfield(cls, x: Tensor, y: Tensor, t: Tensor, eps=_EPS) -> tuple[Tensor, Tensor]:
        dist = cls.dist(x, y, eps).unsqueeze(-1).detach()
        xt = cls.interpolate(x, y, t, eps)
        ux = cls.proj_u(x - xt, xt)
        ux_norm = ux.norm(dim=-1, keepdim=True)
        uy = cls.proj_u(y - xt, xt)
        vf = dist * torch.where(ux_norm > eps, -ux / ux_norm, F.normalize(uy, dim=-1))
        return xt, vf

    @classmethod
    def rand(cls, *sizes, device=None, eps=_EPS) -> Tensor:
        x = torch.randn(*sizes, device=device)
        return F.normalize(x, p=2, dim=-1)

    @classmethod
    def cov_deriv(cls, u: Tensor, du: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        r""":math:`\nabla_v u = \dot u + \langle u, v\rangle x`"""
        return du + (u * v).sum(dim=-1, keepdim=True) * x

    @classmethod
    def dexp_x(cls, x: Tensor, u: Tensor, du: Tensor, eps=_EPS) -> Tensor:
        r""":math:`d(\exp_x)_u(v) = v_\parallel\cos\|u\| + \sinc \|u\| (v_\perp - \langle u, v\rangle x)`"""
        theta = torch.norm(u, dim=-1, keepdim=True)
        inner = (u * du).sum(dim=-1, keepdim=True)
        cos, sinc = torch.cos(theta), torch.sinc(theta / torch.pi)
        u_para = inner / (theta ** 2).clamp(min=eps) * u
        return cos * u_para + sinc * (du - u_para) - inner * sinc * x

    @classmethod
    def dexp_u(cls, x: Tensor, u: Tensor, dx: Tensor, eps=_EPS) -> Tensor:
        r""":math:`d(\exp u)_x(v) = v \cos\|u\| - \sinc \|u\| \langle u, v\rangle x`"""
        theta = torch.norm(u, dim=-1, keepdim=True)
        inner = (u * dx).sum(dim=-1, keepdim=True)
        return torch.cos(theta) * dx - inner * torch.sinc(theta / torch.pi) * x


@register_manifold('so3')
class SO3(Manifold):
    """SO(3) manifold implementation with batch support."""

    def __init__(self, ndim, **kwargs):
        super().__init__(ndim, **kwargs)
        assert self.ndim == 3, "SO(3) manifold only supports 3D rotations."

    @staticmethod
    def hat(v: Tensor) -> Tensor:
        """
        Convert vector to skew-symmetric matrix.
        :param v: vector, Tensor of shape (..., 3)
        :return: skew-symmetric matrix, Tensor of shape (..., 3, 3)
        """
        batch_shape = v.shape[:-1]
        v = v.reshape(-1, 3)
        hat_v = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
        hat_v[..., 0, 1] = -v[..., 2]
        hat_v[..., 0, 2] = v[..., 1]
        hat_v[..., 1, 0] = v[..., 2]
        hat_v[..., 1, 2] = -v[..., 0]
        hat_v[..., 2, 0] = -v[..., 1]
        hat_v[..., 2, 1] = v[..., 0]
        return hat_v.view(*batch_shape, 3, 3)

    @staticmethod
    def vee(m: Tensor) -> Tensor:
        """
        Convert skew-symmetric matrix to vector.
        :param m: skew-symmetric matrix, Tensor of shape (..., 3, 3)
        :return: vector, Tensor of shape (..., 3)
        """
        return torch.stack([m[..., 2, 1], m[..., 0, 2], m[..., 1, 0]], dim=-1)

    @classmethod
    def lie_exp(cls, x: Tensor, eps=_EPS):
        """
        Convert rotation vector to rotation matrix.
        :param x: rotation vector, Tensor of shape (..., 3)
        :param eps: small float to avoid instability
        :return: rotation matrix, Tensor of shape (..., 3, 3)
        """
        theta = torch.norm(x, dim=-1, keepdim=True).unsqueeze(-1)
        theta2 = theta ** 2
        u_hat = cls.hat(x)
        I = torch.eye(3, device=x.device, dtype=x.dtype)[(None,) * (x.dim() - 1)]
        mask = theta < eps
        coef1 = torch.where(mask, 1.0 - theta2 / 6, torch.sin(theta) / theta)
        coef2 = torch.where(mask, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2)
        return I + coef1 * u_hat + coef2 * (u_hat @ u_hat)

    @classmethod
    def lie_log(cls, x: Tensor, eps=_EPS):
        """
        Convert rotation matrix to rotation vector
        :param x: rotation matrix, Tensor of shape (..., 3, 3)
        :param eps: small float to avoid instability
        :return: rotation vector, Tensor of shape (..., 3)
        """
        axis = cls.vee(x - x.transpose(-2, -1))
        cos_theta = (torch.einsum("...ii", x) - 1) / 2
        sin_theta = axis.norm(dim=-1) / 2
        theta = torch.atan2(sin_theta, cos_theta)
        coeff = torch.where(theta < eps, 0.5 + theta ** 2 / 12, theta / (2 * torch.sin(theta)))
        rotvec = coeff.unsqueeze(-1) * axis
        mask_pi = torch.isclose(theta, torch.full_like(theta, np.pi), atol=1e-3)
        if mask_pi.any():
            x_pi = x[mask_pi]
            theta_pi = theta[mask_pi].unsqueeze(-1)
            # Implementation from geomstats
            eye = torch.eye(3, device=x.device, dtype=x.dtype).unsqueeze(0).expand_as(x_pi)
            outer = (eye + x_pi) / 2
            outer = outer + (torch.relu(outer) - outer) * eye
            axis_pi = torch.diagonal(outer, dim1=-2, dim2=-1).clamp(min=1e-8).sqrt()
            sign_idx = torch.argmax(torch.norm(outer, dim=-1, keepdim=True), dim=-2, keepdim=True)
            sign_line = torch.take_along_dim(outer, dim=-2, indices=sign_idx).squeeze(-2)
            rotvec_pi = axis_pi * theta_pi * torch.sign(sign_line)

            rotvec.masked_scatter_(mask_pi.unsqueeze(-1), rotvec_pi)
        return rotvec

    @classmethod
    def apply_rotvec(cls, x: Tensor, u: Tensor, eps=_EPS) -> Tensor:
        """
        Apply the rotation vector on a vector using Rodrigues' rotation formula.
        :param x: rotation vector, Tensor of shape (..., 3)
        :param u: vector to be rotated, Tensor of shape (..., 3)
        :param eps: small float to avoid instability
        :return: rotated vector, Tensor of shape (..., 3)
        """
        theta = torch.norm(x, dim=-1, p=2, keepdim=True)
        theta2 = theta ** 2
        mask = theta < eps
        sin = torch.where(mask, 1.0 - theta2 / 6, torch.sin(theta) / theta)
        cos = torch.where(mask, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2)
        ad = torch.cross(x, u, dim=-1)
        ad2 = torch.cross(x, ad, dim=-1)
        return u + sin * ad + cos * ad2

    @classmethod
    def exp(cls, x: Tensor, u: Tensor, eps=_EPS) -> Tensor:
        r"""Exponential map :math:`\exp_x(u) = x \Exp(u)`. Will convert final result back to rotation vector.
        :param x: base rotation vector, Tensor of shape (..., 3)
        :param u: angent vector, Tensor of shape (..., 3)
        :param eps: small float to avoid instability
        :return: target rotation vector, Tensor of shape (..., 3)
        """
        x_rot = cls.lie_exp(x, eps)
        u_rot = cls.lie_exp(u, eps)
        return cls.lie_log(x_rot @ u_rot, eps)

    @classmethod
    def log(cls, x: Tensor, y: Tensor, eps=_EPS) -> Tensor:
        r"""Logarithm map :math:`\log_x(y) = \Log(x^\top y)`.
        :param x: base rotation vector of shape (..., 3)
        :param y: target rotation vector of shape (..., 3)
        :param eps: small float to avoid instability
        :return: Tangent vector of shape (..., 3)
        """
        return cls.exp(-x, y, eps)

    @classmethod
    def proj_x(cls, x: Tensor, eps=_EPS) -> Tensor:
        return x

    @classmethod
    def proj_u(cls, u: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        return u

    @classmethod
    def inner(cls, u: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        """Riemannian inner product on SO(3), which coincides with the Euclidean inner product for 3-vectors.
        :param u: tangent vector of shape (..., 3)
        :param v: tangent vector of shape (..., 3)
        :param x: base rotation vector point of shape (..., 3)
        :return: Inner product of shape (...)
        """
        return (u * v).sum(dim=-1)

    @classmethod
    def vecfield(cls, x: Tensor, y: Tensor, t: Tensor, eps=_EPS) -> tuple[Tensor, Tensor]:
        vf = cls.log(x, y, eps)
        xt = cls.exp(x, t.unsqueeze(-1) * vf, eps)
        return xt, vf

    @classmethod
    def rand(cls, *sizes, device=None, eps=_EPS) -> Tensor:
        quat = F.normalize(torch.randn(*sizes[:-1], 4, device=device), dim=-1)
        quat = torch.where(quat[..., :1] < 0, -quat, quat)
        axes = quat[..., :3]
        norm = torch.norm(axes, dim=-1)
        theta = torch.atan2(norm, quat[..., 0]) * 2
        scale = torch.where(theta < eps, 2 + theta ** 2 / 12, theta / torch.sin(theta / 2))
        return scale.unsqueeze(-1) * axes

    @classmethod
    def cov_deriv(cls, u: Tensor, du: Tensor, v: Tensor, x: Tensor, eps=_EPS) -> Tensor:
        r""":math:`\nabla_v u = \dot u + v \times u / 2`"""
        return du + torch.cross(v, u, dim=-1) / 2

    @classmethod
    def left_jac(cls, u: Tensor, v: Tensor, eps=_EPS) -> Tensor:
        r"""Left Jacobian of SO(3) applied to v :math:`J_L(u)v`."""
        theta = torch.norm(u, dim=-1, keepdim=True)
        theta2 = theta ** 2
        mask = theta < eps
        sin = torch.where(mask, 1 / 6 - theta2 / 120, (1 - torch.sinc(theta / np.pi)) / theta2)
        cos = torch.where(mask, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2)
        ad = torch.cross(u, v, dim=-1)
        ad2 = torch.cross(u, ad, dim=-1)
        w = v + cos * ad + sin * ad2
        return w

    @classmethod
    def right_jac(cls, u: Tensor, v: Tensor, eps=_EPS) -> Tensor:
        r"""Right Jacobian of SO(3) applied to v :math:`J_R(u)v`."""
        return cls.left_jac(-u, v, eps)

    @classmethod
    def right_jac_inv(cls, u: Tensor, v: Tensor, eps=_EPS) -> Tensor:
        r"""Right inverse Jacobian of SO(3) applied to v :math:`J^{-1}_R(u)v`."""
        theta = torch.norm(u, dim=-1, keepdim=True)
        theta2 = theta ** 2
        second = torch.where(
            theta < eps,
            1 / 12 + theta2 / 720,
            1 / theta2 - (1 + torch.cos(theta)) / (2 * theta * torch.sin(theta))
        )
        ad = torch.cross(u, v, dim=-1)
        ad2 = torch.cross(u, ad, dim=-1)
        w = v + ad / 2 + ad2 * second
        return w

    @classmethod
    def left_jac_inv(cls, u: Tensor, v: Tensor, eps=_EPS) -> Tensor:
        r"""Left inverse Jacobian of SO(3) applied to v :math:`J^{-1}_L(u)v`."""
        return cls.right_jac_inv(-u, v, eps)

    @classmethod
    def dexp_x(cls, x: Tensor, u: Tensor, du: Tensor, eps=_EPS) -> Tensor:
        r""":math:`d(\exp_x)_u(v) = J_R(u)v`"""
        return cls.right_jac(u, du, eps)

    @classmethod
    def dexp_u(cls, x: Tensor, u: Tensor, dx: Tensor, eps=_EPS) -> Tensor:
        r""":math:`d(\exp u)_x(v) = R_u^\top v - J_R(u) (v \times u) / 2`"""
        dexp = cls.apply_rotvec(-u, dx, eps)
        return dexp - cls.dexp_x(x, u, torch.cross(dx, u, dim=-1) / 2, eps)
