import math
import torch
import warnings
import torch.nn as nn
from typing import List
from torchmetrics import Metric
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1)
        self.nonlinearity = nn.SiLU()
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states):
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv2(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        return hidden_states


class DeconvUpsampler(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_deconv):
        super().__init__()
        self.up_in = UpSampleBlock(in_channels, hidden_channels, stride=2)
        self.up_blocks = nn.ModuleList([UpSampleBlock(hidden_channels, hidden_channels, stride=2) for i in range(num_deconv-1)])
        self.out_conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.up_in(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.out_conv(x)
        return x


class FourierEmbedder(nn.Module):
    def __init__(self, num_freqs=8, logspace=True, input_dim=3):
        super().__init__()
        self.num_freqs = num_freqs
        self.out_dim = input_dim * (self.num_freqs * 2 + 1)

        if logspace:
            frequencies = 2.0 ** torch.arange(num_freqs)
        else:
            frequencies = torch.linspace(1.0, 2.0 ** (num_freqs - 1), num_freqs)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x):
        embed = (x[..., None].contiguous() * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((x, embed.sin(), embed.cos()), dim=-1)


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, feat_dim=1):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self):
        x = self.mean + self.std * torch.randn_like(self.mean)
        return x

    def kl(self, other=None, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.mean(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=dims)
            else:
                return 0.5 * torch.mean(torch.pow(self.mean - other.mean, 2) / other.var + self.var / other.var - 1.0 - self.logvar + other.logvar, dim=dims)

    def nll(self, sample, dims=(1, 2)):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class BCEWithLogitsMetric(Metric):
    def __init__(self):
        super().__init__()
        self.occ_loss = torch.nn.BCEWithLogitsLoss(reduce=False)
        self.add_state("accum", default=torch.tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        
        loss_bce = self.occ_loss(preds, target).mean(-1)

        self.accum += loss_bce.sum()
        self.total += target.shape[0]

    def compute(self):
        return self.accum / self.total


"""
code copied from lightning-bolts
"""
class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr and
    base_lr followed by a cosine annealing schedule between base_lr and eta_min.

    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.

    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler; please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            ) * (group["lr"] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]
