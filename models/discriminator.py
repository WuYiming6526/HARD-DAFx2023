import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Conv2dDiscriminator(nn.Module):
    def __init__(self, in_width, **kwargs):
        super().__init__()
        n_layers = 5
        net = [spectral_norm(nn.Conv2d(in_width, 16, 1)), nn.LeakyReLU(0.2)]

        for i in range(n_layers):
            net.append(
                spectral_norm(
                    nn.Conv2d(
                        min(512, 16 * 2**i), min(512, 16 * 2 ** (i + 1)), 3, stride=2
                    )
                )
            )
            # net.append(nn.LeakyReLU(.2))
            # net.append(weight_norm(nn.Conv2d(min(512, 16*2**(i+1)),
            #                                 min(512, 16*2**(i+1)),
            #                                 3,
            #                                 stride=2)))
            net.append(nn.LeakyReLU(0.2))
        net.append(spectral_norm(nn.Conv2d(min(512, 16 * 2 ** (i + 1)), 1, 1)))
        self.net = nn.ModuleList(net)

    def forward(self, x: torch.Tensor):
        feature = []
        if x.ndim == 3:
            x = x.unsqueeze(1)
        for layer in self.net:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                feature.append(x)
        return [feature]
