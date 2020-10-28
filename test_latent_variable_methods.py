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

from latent_mapping import init_noises, latent_mappings, argmax_onehot
from spigot.algorithms.krucker import project_onto_knapsack_constraint_batch

from sparsemax import Sparsemax

logger = logging.getLogger(__file__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


eps = 1e-12


def entropy(x):
    return - (x * torch.log(x + eps)).sum(dim=-1).mean()


def kl_divergence(qs, ps, mask=None):
    outputs = torch.sum(ps * (torch.log(ps + eps) - torch.log(qs + eps)), dim=-1)

    if mask is None:
        return outputs.mean()
    return (outputs * mask.float().unsqueeze(-1)).mean()


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
    uniform_center=False,
):
    # generate cluster centers
    # if centers is None:
    #     centers = cluster_std * torch.randn(n_clusters, n_features)
    if centers is None:
        if uniform_center:
            centers = cluster_std * (torch.rand(n_clusters, n_features) - 0.5)
        else:
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


def Feedforward(in_dim, out_dim, num_layers=1, hidden_dim=None):
    assert num_layers > 0
    if num_layers == 1:
        return torch.nn.Linear(in_dim, out_dim)

    layers = []
    in_dim_ = in_dim
    out_dim_ = hidden_dim or in_dim
    for i in range(num_layers):
        layers.append(torch.nn.Linear(in_dim_, out_dim_))
        in_dim_ = out_dim_
        if i < num_layers - 1:
            layers.append(torch.nn.ReLU())
        if i == num_layers - 2:
            out_dim_ = out_dim
    return torch.nn.Sequential(*layers)


class Net(torch.nn.Module):
    def __init__(self, K, dim_X, num_layers=1, mapping_fun="softmax", gumbel=False):
        super().__init__()
        self.encoder = Feedforward(dim_X, K, num_layers=num_layers)
        self.latent_mapping = latent_mappings[mapping_fun]
        self.decoder = torch.nn.Bilinear(K, dim_X, 1, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(K, dtype=torch.float))
        self.gumbel = gumbel

    def forward(self, x, y, z):
        s = self.encoder(x)
        if self.training and self.gumbel:
            s += gumbel_noise_like(s)
        # s = torch.softmax(s, dim=-1)
        z_hat = self.latent_mapping(s)
        z_hat_index = torch.argmax(z_hat, dim=-1)
        y_hat = self.decoder(z_hat, x) + torch.matmul(z_hat, self.decoder_bias.unsqueeze(dim=1))
        y_hat = y_hat.squeeze(dim=-1)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        accuracy = (torch.sign(y_hat) == y).float().mean()
        
        return y_hat, z_hat_index, loss, accuracy


class DiagnosticNet(torch.nn.Module):
    def __init__(self, K, dim_X, num_layers=1, mapping_fun="spigot", gumbel=False):
        super().__init__()
        self.K = K
        self.encoder = Feedforward(dim_X, K, num_layers=num_layers)
        assert mapping_fun in ["spigot", "ste", "spigot_ce", "oracle"]
        self.mapping_fun = mapping_fun
        self.decoder = torch.nn.Bilinear(K, dim_X, 1, bias=False)
        self.decoder_bias = torch.nn.Parameter(torch.zeros(K, dtype=torch.float))
        self.gumbel = gumbel
        self.epoch = 0

    def eval(self):
        super().eval()
        self.epoch += 1

    def forward(self, x, y, z):
        s = self.encoder(x)
        if self.training and self.gumbel:
            s += gumbel_noise_like(s)
        z_hat = argmax_onehot(s)
        z_hat_index = torch.argmax(z_hat, dim=-1)

        if self.mapping_fun == "oracle":
            z_tilde = torch.nn.functional.one_hot(z, self.K).float().requires_grad_(True)
            z_hat_index = z
        else:
            z_tilde = z_hat.clone().detach().requires_grad_(True)

        y_hat = self.decoder(z_tilde, x) + torch.matmul(z_tilde, self.decoder_bias.unsqueeze(dim=1))
        y_hat = y_hat.squeeze(dim=-1)

        loss = torch.nn.functional.mse_loss(y_hat, y)
        accuracy = (torch.sign(y_hat) == y).float().mean()

        torch.autograd.backward(loss)

        norm = torch.norm(z_tilde.grad, dim=-1)
        scale = torch.ones_like(norm)
        scale[norm > 1.0] = 1.0 / norm[norm > 1.0]

        if self.mapping_fun == "spigot":
            mu = Sparsemax(dim=-1)(
                z_tilde - scale.unsqueeze(dim=-1) * z_tilde.grad
            )
            torch.autograd.backward(s, z_hat - mu)
        elif self.mapping_fun == "ste":
            torch.autograd.backward(s, z_tilde.grad)
        elif self.mapping_fun == "oracle":
            # just to fill some grads
            torch.autograd.backward(s, torch.zeros_like(z_tilde.grad))
        elif self.mapping_fun == "spigot_ce":
            ss = torch.softmax(s, dim=-1)
            mu = Sparsemax(dim=-1)(
                ss - scale.unsqueeze(dim=-1) * z_tilde.grad
            )
            torch.autograd.backward(s, ss - mu)
        else:
            assert False

        return y_hat, z_hat_index, loss, accuracy


def train(
    net,
    optimizer,
    train_xs,
    train_zs,
    train_ys,
    valid_xs,
    valid_zs,
    valid_ys,
    epochs=10000,
    backward=True,
    clip_grad=None,
    report_epoch=100,
):

    train_zs_cpu = train_zs.cpu()
    valid_zs_cpu = valid_zs.cpu()

    for epoch in range(epochs):

        net.train()
        net.zero_grad()
        _, z_hat, loss, accuracy = net(train_xs, train_ys, train_zs)

        if not isinstance(net, DiagnosticNet):
            loss.backward()

        if clip_grad is not None:
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_grad)

        optimizer.step()

        net.eval()
        if epoch % report_epoch == 0:

            loss_train = loss.cpu().item()
            accuracy_train = accuracy.cpu().item()
            v_measure_train = v_measure_score(train_zs_cpu, z_hat.detach().cpu())

            _, z_hat, loss, accuracy = net(valid_xs, valid_ys, valid_zs)

            loss_valid = loss.cpu().item()
            accuracy_valid = accuracy.cpu().item()
            v_measure_valid = v_measure_score(valid_zs_cpu, z_hat.detach().cpu())


            logger.info(
                f"epoch: {epoch} loss (train): {loss_train:.4f}, loss (dev): {loss_valid:.4f}, "
                f"acc. (train): {accuracy_train:.4f}, acc. (valid): {accuracy_valid:.4f} "
                f"v-mes. (train): {v_measure_train:.4f}, v-mes. (valid): {v_measure_valid:.4f}"
            )

            logits = net.encoder(train_xs)
            ent = entropy(torch.softmax(logits, dim=-1)).item()
            maxv = logits.abs().max().item()
            logger.info(f'entropy {ent:.4f}, maxv {maxv:.4f}')
            # IPython.embed(colors='neutral')


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
                cfg.train_size, cfg.latent_dim, cfg.K, device=device, # uniform_center=True
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

            torch.manual_seed(seed + 40)
            numpy.random.seed(seed + 40)

            net = (DiagnosticNet if cfg.diagnostic else Net)(
                K=cfg.K,
                dim_X=cfg.latent_dim,
                num_layers=cfg.num_layers,
                mapping_fun=cfg.latent_mapping,
                gumbel=cfg.gumbel,
            ).to(device)

            if cfg.oracle_decoder:
                net.decoder.weight = torch.nn.Parameter(W.unsqueeze(0).to(device))
                net.decoder_bias = torch.nn.Parameter(-b.to(device))

                optimizer = torch.optim.Adam(
                    [
                        {"params": net.encoder.parameters()},
                        {"params": list(net.decoder.parameters()) + [net.decoder_bias], "lr": 0.0},
                    ],
                    lr=cfg.lr,
                )
            else:
                optimizer = torch.optim.Adam(net.parameters(), lr=cfg.lr)

            logger.info(
                f"seed: {seed}, latent mapping: {cfg.latent_mapping}, gumbel: {cfg.gumbel}"
            )
            accuracy, v_measure = train(
                net,
                optimizer,
                train_xs,
                train_zs,
                train_ys,
                valid_xs,
                valid_zs,
                valid_ys,
                epochs=cfg.epochs,
                clip_grad=cfg.clip_grad,
                report_epoch=cfg.report_epoch
            )

            writer.writerow([seed, cfg.latent_mapping, cfg.gumbel, accuracy, v_measure])


if __name__ == "__main__":
    main()
