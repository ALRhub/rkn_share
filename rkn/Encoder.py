import torch
from typing import Tuple

nn = torch.nn


def elup1(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x).where(x < 0.0, x + 1.0)


class Encoder(nn.Module):

    def __init__(self, lod: int, output_normalization: str = "post"):
        """Gaussian Encoder, as described in ICML Paper (if output_normalization=post)
        :param lod: latent observation dim, i.e. output dim of the Encoder mean and var
        :param output_normalization: when to normalize the output:
            - post: after output layer (as described in ICML paper)
            - pre: after last hidden layer, that seems to work as well in most cases but is a bit more principled
            - none: (or any other string) not at all

        """
        super(Encoder, self).__init__()
        self._hidden_layers, size_last_hidden = self._build_hidden_layers()
        assert isinstance(self._hidden_layers, nn.ModuleList), "_build_hidden_layers needs to return a " \
                                                               "torch.nn.ModuleList or else the hidden weights are " \
                                                               "not found by the optimizer"
        self._mean_layer = nn.Linear(in_features=size_last_hidden, out_features=lod)
        self._log_var_layer = nn.Linear(in_features=size_last_hidden, out_features=lod)

        self._output_normalization = output_normalization

    def _build_hidden_layers(self) -> Tuple[nn.ModuleList, int]:
        """
        Builds hidden layers for encoder
        :return: nn.ModuleList of hidden Layers, size of output of last layer
        """
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = obs
        for layer in self._hidden_layers:
            h = layer(h)
        if self._output_normalization.lower() == "pre":
            h = nn.functional.normalize(h, p=2, dim=-1, eps=1e-8)

        mean = self._mean_layer(h)
        if self._output_normalization.lower() == "post":
            mean = nn.functional.normalize(mean, p=2, dim=-1, eps=1e-8)

        log_var = self._log_var_layer(h)
        var = elup1(log_var)
        return mean, var
