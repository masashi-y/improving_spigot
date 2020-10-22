import argparse
import csv
from typing import Optional

import IPython
import numpy
import torch
from sklearn.metrics import v_measure_score
from sparsemax import Sparsemax

from spigot.algorithms.krucker import project_onto_knapsack_constraint_batch

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


class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        _, feature_size = x.size()
        results = torch.nn.functional.one_hot(
            torch.max(x, dim=-1).indices, num_classes=feature_size,
        ).float()

        return results

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class StraightThroughSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        _, feature_size = x.size()
        results = torch.nn.functional.one_hot(
            torch.max(x, dim=-1).indices, num_classes=feature_size,
        ).float()

        ctx.save_for_backward(x)
        return results

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors

        with torch.enable_grad():
            (grad,) = torch.autograd.grad(torch.softmax(x, dim=-1), x, grad_output)

        return grad_output, None


class SPIGOT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):

        _, feature_size = x.size()
        results = torch.nn.functional.one_hot(
            torch.max(x, dim=-1).indices, num_classes=feature_size,
        ).float()

        ctx.save_for_backward(results)
        return results

    @staticmethod
    def backward(ctx, grad_output):
        (predictions,) = ctx.saved_tensors

        norm = torch.norm(grad_output, dim=-1)
        scale = torch.ones_like(norm)
        cond = norm > 1.0
        scale[cond] = 1.0 / norm[cond]

        target = - predictions + scale.unsqueeze(dim=-1) * grad_output

        projected = project_onto_knapsack_constraint_batch(target)
        # output = predictions - projected
        output = projected - predictions
        return output
        output_scaled = norm.unsqueeze(dim=-1) * output / (torch.norm(output, dim=-1, keepdim=True) + 1e-12)
        # return output_scaled
        # return torch.max(output, torch.zeros_like(output_scaled))


GAUSSIAN, UNIFORM = None, None


def init_noises(K, device=None):
    global GAUSSIAN, UNIFORM
    GAUSSIAN = torch.randn((K, K), device=device)
    UNIFORM = torch.rand((K, K), device=device)
    print(GAUSSIAN.shape)


class Noise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, noise):

        _, feature_size = x.size()
        results = torch.nn.functional.one_hot(
            torch.max(x, dim=-1).indices, num_classes=feature_size,
        ).float()

        ctx.save_for_backward(noise)
        return results

    @staticmethod
    def backward(ctx, grad_output):
        (noise,) = ctx.saved_tensors
        return torch.matmul(grad_output, noise), None


def make_latent_triples(
    n_samples,
    n_features,
    n_clusters,
    centers=None,
    W=None,
    b=None,
    data_std=0.1,
    cluster_std=1,
    device=None
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


latent_mappings = {
    "spigot": SPIGOT.apply,
    # "sparsemax": Sparsemax(dim=-1),
    # "softmax": torch.nn.Softmax(dim=-1),
    # "ste": StraightThrough.apply,
    # "sts": StraightThroughSoftmax.apply,
    # "gaussian": lambda x: Noise.apply(x, GAUSSIAN),
    # "uniform": lambda x: Noise.apply(x, UNIFORM),
}


class Net(torch.nn.Module):
    def __init__(self, K, dim_X, mapping_fun="softmax", gumbel=False):
        super().__init__()
        self.encoder = torch.nn.Linear(dim_X, K)
        self.latent_mapping = latent_mappings[mapping_fun]
        self.decoder = torch.nn.Bilinear(K, dim_X, 1)
        self.gumbel = gumbel

    def forward(self, x):
        s = self.encoder(x)
        if self.training and self.gumbel:
            s += gumbel_noise_like(s)
        z_hat = self.latent_mapping(s)
        z_hat_index = torch.argmax(z_hat, dim=-1)
        y_hat = self.decoder(z_hat, x).squeeze(dim=-1)
        return y_hat, z_hat_index


def train(
    net, train_xs, train_zs, train_ys, valid_xs, valid_zs, valid_ys,
):

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    train_zs_cpu = train_zs.cpu()
    valid_zs_cpu = valid_zs.cpu()

    for epoch in range(args.epochs):

        net.train()

        predictions, z_hat = net(train_xs)

        loss = torch.nn.functional.mse_loss(predictions, train_ys)

        accuracy = (torch.sign(predictions) == torch.sign(train_ys)).float().mean()

        net.zero_grad()
        loss.backward()

        optimizer.step()

        loss_train = loss.cpu().item()
        accuracy_train = accuracy.cpu().item()
        v_measure_train = v_measure_score(train_zs_cpu, z_hat.detach().cpu())

        net.eval()
        with torch.no_grad():
            predictions, z_hat = net(valid_xs)

        loss_valid = torch.nn.functional.mse_loss(predictions, valid_ys).cpu().item()
        accuracy_valid = (
            (torch.sign(predictions) == torch.sign(valid_ys)).float().mean().item()
        )

        v_measure_valid = v_measure_score(valid_zs_cpu, z_hat.detach().cpu())

        if epoch % 100 == 0:
            print(
                f"epoch: {epoch} loss (train): {loss_train:.4f}, loss (dev): {loss_valid:.4f}, "
                f"acc. (train): {accuracy_train:.4f}, acc. (valid): {accuracy_valid:.4f} "
                f"v-mes. (train): {v_measure_train:.4f}, v-mes. (valid): {v_measure_valid:.4f}"
            )

    return accuracy_valid, v_measure_valid


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--latent-dim", type=int, default=100)
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--valid-size", type=int, default=1000)
    parser.add_argument("--epoch", type=int, default=10000)

    args = parser.parse_args()
    device = torch.device("cpu" if args.device < 0 else f"cuda:{args.device}")

    init_noises(args.K, device)

    print("latent dimension", args.latent_dim)
    print("dataset size train/valid", args.train_size, args.valid_size)
    print("epoch", args.epochs)

    with open("results_K10_spigot.csv", "w") as f:
        writer = csv.writer(f)

        for seed in range(10):
            # seed += 40

            torch.manual_seed(seed)
            numpy.random.seed(seed)

            train_xs, train_ys, train_zs, centers, W, b = make_latent_triples(
                args.train_size, args.latent_dim, args.K, device=device
            )
            valid_xs, valid_ys, valid_zs, *_ = make_latent_triples(
                args.valid_size, args.latent_dim, args.K, centers=centers, W=W, b=b, device=device
            )

            for j in range(5):

                for latent_mapping in latent_mappings.keys():
                    for gumbel in (True,):
                        torch.manual_seed(seed + j + 40)
                        numpy.random.seed(seed + j + 40)

                        net = Net(K=args.K, dim_X=args.latent_dim, mapping_fun=latent_mapping, gumbel=gumbel).to(device)

                        with torch.autograd.set_detect_anomaly(False):
                            print(seed, j, latent_mapping, gumbel)
                            accuracy, v_measure = train(
                                net, train_xs, train_zs, train_ys, valid_xs, valid_zs, valid_ys
                            )

                        writer.writerow([seed, j, latent_mapping, gumbel, accuracy, v_measure])
