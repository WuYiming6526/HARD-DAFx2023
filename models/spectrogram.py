import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class MelSpecEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        hop_size: int,
        sample_rate: int,
        n_fft: int = 2048,
        power: float = 1.0,
        f_max: int = None,
        mono: bool = False,
        supress_channel: bool = True,
        center: bool = False,
        log_scale: bool = True,
        **kwargs
    ):
        super().__init__()
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_size,
            n_mels=n_mels,
            f_max=f_max,
            power=power,
            center=center,
            mel_scale="slaney",
            norm="slaney",
        )
        self.n_fft = n_fft
        self.hop_length = hop_size
        self.supress_channel = supress_channel
        self.mono = mono
        self.log_scale = log_scale
        if center:
            self.pre_padding = nn.Identity()
        else:
            pad = (self.n_fft - self.hop_length) // 2
            self.pre_padding = nn.ReflectionPad1d((pad, pad))

        inv_melfilter = torch.linalg.pinv(self.melspec.mel_scale.fb.T).clamp(min=0.0)
        self.register_buffer("inv_melfilter", inv_melfilter[None, None, ...])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mono and x.shape[1] > 1:
            x = torch.mean(x, dim=1, keepdim=True)
        x = self.pre_padding(x)
        z = self.melspec(x)
        if self.log_scale:
            z = torch.log(torch.clip(z, min=1e-5))
        if self.supress_channel:
            z = torch.flatten(z, start_dim=1, end_dim=2)
        return z
