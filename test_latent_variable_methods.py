import argparse
import csv
import logging
from typing import Optional

import hydra
import IPython
import numpy
import torch
from omegaconf import OmegaConf
from sklearn.metrics import v_measure_score

from latent_mapping import init_noises, latent_mappings
from spigot.algorithms.krucker import project_onto_knapsack_constraint_batch
from spigot.optim import GradientDescentOptimizer

logger = logging.getLogger(__file__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def gumbel_noise(
    shape: torch.Size,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float,
    eps: float = 1e-12,
) -> torch.FloatTensor:
    uniform = torch.rand(shape).to(device)

    return -torch.log(-torch.log(uniform + eps) + eps)


def gumbel_noise_like(tensor: torch.FloatTensor) -> torch.FloatTensor:
    return gumbel_noise(shape=tensor.size(), device=tensor.device, dtype=tensor.dtype)


def make_latent_triples(
    n_samples,
    n_features,
    n_clusters,
    centers=None,
    W=None,
    b=None,
    data_std=0.1,
    cluster_std=1,
    device=None,
):
    # generate cluster centers
    if centers is None:
        centers = cluster_std * torch.randn(n_clusters, n_features)

    # generate a linear model for each cluster
    if W is None:
        W = torch.randn(n_clusters, n_features)

    # draw cluster assignments
    z = torch.randint(low=0, high=n_clusters, size=(n_samples,))

    # draw data X
    c_ = centers[z]
    X = c_ + data_std * torch.randn(n_samples, n_features)

    # choose linear model to use for each sample
    W_ = W[z]

    # compute true label y
    y_score = (W_ * X).sum(dim=-1)

    # pick a threshold for each class
    # (note: this is done like this to ensure there are always roughly balanced
    # positive and negative samples in each class)
    if b is None:
        b = torch.zeros(n_clusters)
    for c in range(n_clusters):
        b[c] = y_score[z == c].mean()

    y = torch.sign(y_score - b[z])

    def to(*arrs):
        return tuple(arr.to(device) for arr in arrs)

    return to(X, y, z) + (centers, W, b)


class Net(torch.nn.Module):
    def __init__(self, K, dim_X, mapping_fun="softmax", gumbel=False):
        super().__init__()
        self.encoder = torch.nn.Linear(dim_X, K)
        self.latent_mapping = latent_mappings[mapping_fun]
        self.decoder = torch.nn.Bilinear(K, dim_X, 1)
        self.gumbel = gumbel

    def forward(self, x, y):
        s = self.encoder(x)
        if self.training and self.gumbel:
            s += gumbel_noise_like(s)
        z_hat = self.latent_mapping(s)
        z_hat_index = torch.argmax(z_hat, dim=-1)
        y_hat = self.decoder(z_hat, x).squeeze(dim=-1)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        accuracy = (torch.sign(y_hat) == y).float().mean()

        return y_hat, z_hat_index, loss, accuracy


class Net2(torch.nn.Module):
    def __init__(self, K, dim_X, mapping_fun="softmax", gumbel=False):
        super().__init__()
        self.encoder = torch.nn.Linear(dim_X, K)
        self.latent_mapping = latent_mappings[mapping_fun]
        self.decoder = torch.nn.Bilinear(K, dim_X, 1)
        self.gumbel = gumbel

    def forward(self, x, y):
        s = self.encoder(x)
        if self.training and self.gumbel:
            s += gumbel_noise_like(s)
        z_hat = self.latent_mapping(s)
        z_hat_index = torch.argmax(z_hat, dim=-1)

        z_tilde = z_hat.clone().detach().requires_grad_(True)
        y_hat = self.decoder(z_tilde, x).squeeze(dim=-1)
        loss = torch.nn.functional.mse_loss(y_hat, y)

        torch.autograd.backward(loss, create_graph=False)

        norm = torch.norm(z_tilde.grad, dim=-1)
        scale = torch.ones_like(norm)
        scale[norm > 1.0] = 1.0 / norm[norm > 1.0]

        z_tilde = project_onto_knapsack_constraint_batch(
            -z_tilde + scale.unsqueeze(dim=-1) * z_tilde.grad
        )

        torch.autograd.backward(
            s, -z_hat.clone().detach() + z_tilde, create_graph=False
        )

        accuracy = (torch.sign(y_hat) == y).float().mean()

        return y_hat, z_hat_index, loss, accuracy


def train(
    net,
    train_xs,
    train_zs,
    train_ys,
    valid_xs,
    valid_zs,
    valid_ys,
    lr=0.001,
    epochs=10000,
):

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train_zs_cpu = train_zs.cpu()
    valid_zs_cpu = valid_zs.cpu()

    for epoch in range(epochs):

        net.train()

        net.zero_grad()
        _, z_hat, loss, accuracy = net(train_xs, train_ys)

        loss.backward()

        optimizer.step()

        loss_train = loss.cpu().item()
        accuracy_train = accuracy.cpu().item()
        v_measure_train = v_measure_score(train_zs_cpu, z_hat.detach().cpu())

        net.eval()
        with torch.no_grad():
            _, z_hat, loss, accuracy = net(valid_xs, valid_ys)

        loss_valid = loss.cpu().item()
        accuracy_valid = accuracy.cpu().item()

        v_measure_valid = v_measure_score(valid_zs_cpu, z_hat.detach().cpu())

        if epoch % 100 == 0:
            logger.info(
                f"epoch: {epoch} loss (train): {loss_train:.4f}, loss (dev): {loss_valid:.4f}, "
                f"acc. (train): {accuracy_train:.4f}, acc. (valid): {accuracy_valid:.4f} "
                f"v-mes. (train): {v_measure_train:.4f}, v-mes. (valid): {v_measure_valid:.4f}"
            )

    return accuracy_valid, v_measure_valid


@hydra.main(config_name="config")
def main(cfg):
    logger.info(OmegaConf.to_yaml(cfg))

    device = torch.device("cpu" if cfg.device < 0 else f"cuda:{cfg.device}")

    init_noises(cfg.K, device)

    logger.info(f"latent dimension: {cfg.latent_dim}")
    logger.info(f"dataset size train/valid: {cfg.train_size}/{cfg.valid_size}")
    logger.info(f"epoch: {cfg.epochs}")

    with open("results.csv", "w") as f:
        writer = csv.writer(f)

        for seed in range(cfg.seed, cfg.seed + 10):

            torch.manual_seed(seed)
            numpy.random.seed(seed)

            train_xs, train_ys, train_zs, centers, W, b = make_latent_triples(
                cfg.train_size, cfg.latent_dim, cfg.K, device=device
            )
            valid_xs, valid_ys, valid_zs, *_ = make_latent_triples(
                cfg.valid_size,
                cfg.latent_dim,
                cfg.K,
                centers=centers,
                W=W,
                b=b,
                device=device,
            )

            torch.manual_seed(seed)
            numpy.random.seed(seed)

            net = Net(
                K=cfg.K,
                dim_X=cfg.latent_dim,
                mapping_fun=cfg.latent_mapping,
                gumbel=cfg.gumbel,
            ).to(device)

            logger.info(
                f"seed: {seed}, latent mapping: {cfg.latent_mapping}, gumbel: {cfg.gumbel}"
            )
            accuracy, v_measure = train(
                net,
                train_xs,
                train_zs,
                train_ys,
                valid_xs,
                valid_zs,
                valid_ys,
                lr=cfg.lr,
                epochs=cfg.epochs,
            )

            writer.writerow([seed, cfg.latent_mapping, cfg.gumbel, accuracy, v_measure])


if __name__ == "__main__":
    main()
