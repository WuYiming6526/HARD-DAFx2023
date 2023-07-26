import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
#from models import hpss

PAD_MODE = "reflect"


class ResConv2d(nn.Module):
    def __init__(self, channels, kernel_size,
                 autoregressive=False, complex=False, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.fwd = nn.Sequential(
            weight_norm(nn.Conv2d(
                channels, channels, kernel_size,
                padding=(padding, padding*2) if autoregressive else padding,
                padding_mode=PAD_MODE,
                dtype=torch.complex64 if complex else None,
                groups=groups)),
            nn.LeakyReLU(0.2),
            weight_norm(nn.Conv2d(
                channels, channels, kernel_size,
                padding=(padding, padding*2) if autoregressive else padding,
                padding_mode=PAD_MODE,
                dtype=torch.complex64 if complex else None,
                groups=groups)),
            nn.LeakyReLU(0.2))

    def forward(self, x):
        y = self.fwd(x)
        return x + y


class Conv2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 down_scale=None, up_scale=None,
                 fix_time=False, upsample_type="tconv",
                 autoregressive=False,
                 complex=False,
                 groups=1):
        super().__init__()
        padding = kernel_size // 2
        module_list = [
            weight_norm(nn.Conv2d(
                in_channels, out_channels,
                kernel_size,
                padding=(padding, padding*2) if autoregressive else padding,
                padding_mode=PAD_MODE,
                dtype=torch.complex64 if complex else None,
                groups=groups)),
            nn.LeakyReLU(0.2)]
        if isinstance(down_scale, int):
            if fix_time:
                module_list.append(weight_norm(nn.Conv2d(
                    out_channels, out_channels,
                    kernel_size=(down_scale, kernel_size),
                    stride=(down_scale, 1),
                    padding=(0, padding*2) if autoregressive else (0, padding),
                    padding_mode=PAD_MODE,
                    dtype=torch.complex64 if complex else None,
                    groups=groups
                )))
            else:
                module_list.append(weight_norm(nn.Conv2d(
                    out_channels, out_channels,
                    kernel_size=down_scale,
                    stride=down_scale,
                    padding_mode=PAD_MODE,
                    dtype=torch.complex64 if complex else None,
                    groups=groups)))
        if isinstance(up_scale, int):
            if upsample_type == "tconv":
                if fix_time:
                    module_list.append(weight_norm(nn.ConvTranspose2d(
                        out_channels, out_channels,
                        kernel_size=(up_scale, kernel_size),
                        stride=(up_scale, 1),
                        padding=(
                            0, padding*2) if autoregressive else (0, padding),
                        dtype=torch.complex64 if complex else None,
                        groups=groups
                    )))
                else:
                    module_list.append(weight_norm(nn.ConvTranspose2d(
                        out_channels, out_channels,
                        kernel_size=up_scale,
                        stride=up_scale,
                        dtype=torch.complex64 if complex else None,
                        groups=groups
                    )))
            elif upsample_type == "nearest":
                if fix_time:
                    module_list.append(
                        nn.UpsamplingNearest2d(scale_factor=(up_scale, 1)))
                    module_list.append(weight_norm(
                        nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                                  padding="same",
                                  padding_mode=PAD_MODE,
                                  dtype=torch.complex64 if complex else None,
                                  groups=groups)))
                else:
                    module_list.append(
                        nn.UpsamplingNearest2d(scale_factor=up_scale))
                    module_list.append(weight_norm(
                        nn.Conv2d(out_channels, out_channels, kernel_size, 1,
                                  padding="same",
                                  padding_mode=PAD_MODE,
                                  dtype=torch.complex64 if complex else None,
                                  groups=groups)))
        self.fwd = nn.Sequential(*module_list)

    def forward(self, x):
        return self.fwd(x)


class ResConv2DEncoder(nn.Module):
    def __init__(
            self,
            audio_channels: int,
            emb_width: int,
            kernel_size: int,
            n_downsample: int = None,
            width_up_bound: int = 256,
            fix_time: bool = False,
            down_scales: list[int] = None,
            autoregressive: bool = False,
            grouped: bool = False
    ):
        super().__init__()
        down_scales = down_scales or ([2] * n_downsample)
        n_downsample = len(down_scales)
        groups = audio_channels if grouped else 1
        self.conv_in = nn.Sequential(weight_norm(nn.Conv2d(in_channels=audio_channels,
                                                           out_channels=32,
                                                           kernel_size=kernel_size,
                                                           stride=1,
                                                           padding=kernel_size//2,
                                                           padding_mode=PAD_MODE,
                                                           groups=groups)),
                                     nn.LeakyReLU(0.2))
        fwd_res = []
        for i, scale in enumerate(down_scales):
            in_size = min([width_up_bound, 32*2**i])
            out_size = min([width_up_bound, 64*2**i])
            fwd_res += [Conv2dModule(in_size, out_size, kernel_size,
                                     down_scale=scale, fix_time=fix_time,
                                     autoregressive=autoregressive,
                                     groups=groups),
                        ResConv2d(out_size, kernel_size,
                                  autoregressive, groups=groups),
                        ]
        self.fwd_res = nn.Sequential(*fwd_res)
        self.conv_out = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels=out_size,
                                  out_channels=emb_width,
                                  kernel_size=1,
                                  groups=1)))

    def forward(self, spec):
        h = self.conv_in(spec)
        h = self.fwd_res(h)
        z = self.conv_out(h)
        return z        # N,width,1,T


class Shared2DDisentEncoder(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        encoder_args: dict,
        branches: int = 2,
        branch_depth: int = 3,
    ):
        super().__init__()
        encoder_class = {
            "ResConv2D": ResConv2DEncoder
        }[encoder_name]

        self.shared_encoder = encoder_class(**encoder_args)

        bottlenecks = []
        for _ in range(branches):
            modules = [
                ResConv2d(encoder_args["emb_width"],
                          encoder_args["kernel_size"])
                for _ in range(branch_depth)
            ] + [weight_norm(nn.Conv2d(encoder_args["emb_width"], encoder_args["emb_width"], 1))]
            bottlenecks.append(nn.Sequential(*modules))

        self.branches = nn.ModuleList(bottlenecks)

    def forward(self, x: torch.Tensor):
        h = self.shared_encoder(x)
        # return (bottleneck(h) for bottleneck in self.branches)
        return self.branches[0](h), self.branches[1](h)


class ResConv2DDecoder(nn.Module):
    def __init__(
        self,
        audio_channels: int,
        emb_width: int,
        kernel_size: int,
        n_upsample: int = None,
        width_up_bound: int = 256,
        upsample_type: str = "tconv",
        fix_time: bool = False,
        up_scales: list[int] = None,
        grouped: bool = False
    ):
        super().__init__()
        groups = audio_channels if grouped else 1
        self.in_width = audio_channels
        up_scales = up_scales or ([2] * n_upsample)
        n_upsample = len(up_scales)
        self.conv_in = nn.Sequential(weight_norm(nn.ConvTranspose2d(
            emb_width, min(width_up_bound, 64*2**(n_upsample-1)), 3, 1, 1,
            groups=1
        )),
            nn.LeakyReLU(0.2))
        fwd_res = []
        for i, scale in enumerate(up_scales):
            in_size = min(width_up_bound, 64*2**(n_upsample-i-1))
            out_size = min(width_up_bound, 32*2**(n_upsample-i-1))
            fwd_res += [ResConv2d(in_size, kernel_size, groups=groups),
                        Conv2dModule(in_size, out_size, kernel_size,
                                     up_scale=scale,
                                     upsample_type=upsample_type,
                                     fix_time=fix_time,
                                     groups=groups)]
        self.fwd_res = nn.Sequential(*fwd_res)
        self.conv_out = nn.Sequential(
            #Conv2dModule(out_size, out_size, kernel_size, groups=groups),
            nn.Conv2d(in_channels=out_size,
                      out_channels=audio_channels,
                      kernel_size=1,
                      groups=groups)
        )

    def forward(self, z):
        h = self.conv_in(z)
        h = self.fwd_res(h)
        out = self.conv_out(h)
        return out
