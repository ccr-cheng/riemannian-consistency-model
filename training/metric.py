import numpy as np
import torch
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity

from training.manifolds import SO3


def sphere_kl(dataset, predictions, N=90, bandwidth=0.02):
    x = np.linspace(-np.pi, np.pi, N)  # longitude
    y = np.linspace(-np.pi / 2, np.pi / 2, N)  # latitude
    _X, _Y = np.meshgrid(x, y)
    xy = np.vstack([_Y.ravel(), _X.ravel()]).T

    kde_dataset = KernelDensity(
        bandwidth=bandwidth, metric="haversine", kernel="gaussian", algorithm="ball_tree"
    ).fit(dataset.numpy() * np.pi / 180.0)
    logp = kde_dataset.score_samples(xy).reshape(_X.shape)

    kde_predictions = KernelDensity(
        bandwidth=bandwidth, metric="haversine", kernel="gaussian", algorithm="ball_tree"
    ).fit(predictions.numpy() * np.pi / 180.0)
    logq = kde_predictions.score_samples(xy).reshape(_X.shape)
    return (np.exp(logp) * (logp - logq)).sum(-1).mean()


def torus_2d_kl(dataset, predictions, N=180, bandwidth=0.04):
    x = np.linspace(0, 2 * np.pi, N)
    y = np.linspace(0, 2 * np.pi, N)
    _X, _Y = np.meshgrid(x, y)
    xy = np.vstack([_Y.ravel(), _X.ravel()]).T

    one = np.ones((dataset.size(0), 1)) * np.pi * 2
    zero = np.zeros_like(one)
    delta1 = np.hstack([zero, one])
    delta2 = np.hstack([one, zero])

    def get_kernel_density(X):
        expand_X = np.vstack([
            X, X + delta1, X - delta1, X + delta2, X - delta2
        ])
        kde = KernelDensity(
            bandwidth=bandwidth, metric="euclidean", kernel="gaussian", algorithm="auto", atol=1e-5, rtol=1e-7
        ).fit(expand_X)
        logit = kde.score_samples(xy).reshape(_X.shape)
        logit = logit - logsumexp(logit, axis=None) + np.log(N) * 2
        return logit

    logp = get_kernel_density(dataset.numpy())
    logq = get_kernel_density(predictions.numpy())
    return (np.exp(logp) * (logp - logq)).sum()


def torus_nd_kl(dataset, predictions, bandwidth=0.08):
    ndim = dataset.size(-1)
    one = np.ones((dataset.size(0))) * np.pi * 2

    def get_kernel_density(X):
        expand_X = [X]
        for i in range(ndim):
            X_minus, X_plus = X.copy(), X.copy()
            X_minus[:, i] -= one
            X_plus[:, i] += one
            expand_X.extend([X_minus, X_plus])
        expand_X = np.vstack(expand_X)
        kde = KernelDensity(
            bandwidth=bandwidth, metric="euclidean", kernel="gaussian", algorithm="auto", atol=1e-5, rtol=1e-7
        ).fit(expand_X)
        return kde.score_samples(dataset.numpy()) + np.log(2 * ndim + 1)

    logp = get_kernel_density(dataset.numpy())
    logq = get_kernel_density(predictions.numpy())
    return (logp - logq).sum(-1).mean()


def mmd(dataset, prediction, dist2_fn, bandwidth=1., batch_size=4096, unbiased=False):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two distributions.
    :param dataset: The first distribution (ground truth).
    :param prediction: The second distribution (predicted).
    :param dist2_fn: The squared-distance function to use for MMD computation.
    :param bandwidth: The bandwidth parameter for the Gaussian kernel.
    :param batch_size: The number of samples to compute.
    :param unbiased: If True, computes the unbiased MMD.
    :return: The MMD value.
    """

    def cdist(x_batch, y_batch):
        res = 0
        for x in x_batch:
            for y in y_batch:
                x_expand = x.unsqueeze(0).expand(y.size(0), -1, *y.size()[1:])
                y_expand = y.unsqueeze(1).expand(-1, x.size(0), *x.size()[1:])
                kernel = (-dist2_fn(x_expand, y_expand) * bandwidth).exp()
                res += kernel.sum()
        return res * 2

    def pdist(x_batch):
        res = 0
        for i in range(len(x_batch)):
            for j in range(i, len(x_batch)):
                x, y = x_batch[i], x_batch[j]
                x_expand = x.unsqueeze(0).expand(y.size(0), -1, *y.size()[1:])
                y_expand = y.unsqueeze(1).expand(-1, x.size(0), *x.size()[1:])
                kernel = (-dist2_fn(x_expand, y_expand) * bandwidth).exp()
                if i == j:
                    kernel.fill_diagonal_(0 if unbiased else 1)
                else:
                    kernel = kernel * 2
                res += kernel.sum()
        return res

    data_batch = torch.split(dataset, batch_size)
    pred_batch = torch.split(prediction, batch_size)
    a, b, c = pdist(data_batch), cdist(data_batch, pred_batch), pdist(pred_batch)
    n, m = dataset.size(0), prediction.size(0)
    if unbiased:
        mmd_value = a / (n * (n - 1)) - b / (n * m) + c / (m * (m - 1))
        if (mmd_value < 0.0).any():
            print('Warning: MMD value < 0.0, use the biased MMD instead.')
    else:
        assert n == m, "Unbiased MMD requires equal number of samples in both distributions."
        mmd_value = ((a - b + c) / (n * (n - 1))).clamp(min=0)
    return mmd_value.sqrt().item()


def sphere_mmd(dataset, prediction, bandwidth=1., batch_size=4096):
    def dist2(x, y):
        d = torch.acos((x * y).sum(-1).clamp(-1, 1))
        return d ** 2

    return mmd(dataset, prediction, dist2, bandwidth, batch_size)


def torus_mmd(dataset, prediction, bandwidth=1., batch_size=4096):
    def dist2(x, y):
        dx = y - x
        dx = dx % (2 * np.pi)
        dx = torch.where(dx > np.pi, dx - 2 * np.pi, dx)
        return (dx ** 2).sum(-1)

    return mmd(dataset, prediction, dist2, bandwidth, batch_size)


def so3_mmd(dataset, prediction, bandwidth=1., batch_size=4096):
    def dist2(x, y):
        dx = x.transpose(-1, -2) @ y
        trace = torch.diagonal(dx, dim1=-2, dim2=-1).sum(-1)
        cos_theta = (trace - 1) / 2
        theta = torch.acos(cos_theta.clamp(-1, 1))
        return theta ** 2

    return mmd(SO3.lie_exp(dataset), SO3.lie_exp(prediction), dist2, bandwidth, batch_size)
