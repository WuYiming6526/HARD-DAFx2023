import torch
import torch.nn as nn
import torch.nn.functional as F

from models import discriminator, encdec_spec_2d as encdec2d


class VAEGANBase(nn.Module):
    def __init__(
        self,
        discriminator_name: str = "no",
        discriminator_args: dict = None,
        adv_loss: str = "square",
    ):
        super().__init__()
        discriminator_class = {
            "Conv2dDiscriminator": discriminator.Conv2dDiscriminator,
        }
        if discriminator_name != "no":
            self.discriminator = discriminator_class[discriminator_name](
                **discriminator_args
            )
        self.adv_loss = adv_loss

    def rec_loss(self, x, y):
        # return F.mse_loss(y, x)
        # return F.l1_loss(y, x)
        return F.huber_loss(y, x)

    def adversarial_generative_loss(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> torch.Tensor:
        feature_real = self.discriminator(x_real)
        feature_fake = self.discriminator(x_fake)
        loss_gen = 0.0
        feature_matching_distance = 0
        for scale_real, scale_fake in zip(feature_real, feature_fake):
            feature_matching_distance += 10 * sum(
                map(lambda x, y: abs(x - y).mean(), scale_real[:-1], scale_fake[:-1])
            )
            if self.adv_loss == "square":
                loss_gen += (scale_fake[-1] - 1).pow(2).mean()
            else:
                loss_gen += -scale_fake[-1].mean()
        return loss_gen, feature_matching_distance

    def discriminator_loss(
        self, x_real: torch.Tensor, x_fake: torch.Tensor
    ) -> torch.Tensor:
        feature_real = self.discriminator(x_real.detach())
        feature_fake = self.discriminator(x_fake.detach())
        loss_dis = 0
        for scale_real, scale_fake in zip(feature_real, feature_fake):
            score_real, score_fake = scale_real[-1], scale_fake[-1]
            if self.adv_loss == "square":
                loss_dis += ((score_real - 1.0).pow(2) + score_fake.pow(2)).mean()
            else:
                loss_dis += (
                    torch.relu(1 - score_real) + torch.relu(1 + score_fake)
                ).mean()
        return loss_dis


def linear_interpolate(x1: torch.Tensor, x2: torch.Tensor, weight):
    if isinstance(weight, torch.Tensor):
        weight = F.interpolate(weight[None, None, :], size=x1.shape[-1], mode="linear")
    x = x1 * (1 - weight) + x2 * weight
    return x


class PitchDisentangleVAEGAN(VAEGANBase):
    def __init__(
        self,
        encoder_name: str,
        encoder_args: dict,
        decoder_name: str,
        decoder_args: dict,
        discriminator_name: str = "no",
        discriminator_args: dict = None,
        adv_loss: str = "square",
        latent_activation: str = "identity",
        no_shift: bool = False,
    ):
        super().__init__(discriminator_name, discriminator_args, adv_loss)
        encoder_class = {
            "SharedConv2D": encdec2d.Shared2DDisentEncoder,
        }
        decoder_class = {
            "ResConv2DDecoder": encdec2d.ResConv2DDecoder,
        }
        self.encoder = encoder_class[encoder_name](**encoder_args)

        self.decoder = decoder_class[decoder_name](**decoder_args)

        self.latent_activation = {
            "identity": nn.Identity,
            "tanh": nn.Tanh,
        }[latent_activation]()

        self.no_shift = no_shift

    def _to_meanvar(self, z, rsample: bool = True):
        mean, var = torch.chunk(z, 2, 1)
        mean = self.latent_activation(mean)
        var = F.softplus(var)
        if rsample:
            sampled = mean + torch.randn_like(mean) * var
        else:
            sampled = mean
        return mean, var, sampled

    def _encode(self, x, rsample: bool = False, ret_kl: bool = True):
        h_h, h_r = self.encoder(x)
        mean_h, var_h, res_h = self._to_meanvar(h_h, rsample=rsample)
        mean_r, var_r, res_r = self._to_meanvar(h_r, rsample=rsample)
        if rsample:
            z_harmony = res_h
            z_rhythm = res_r
        else:
            z_harmony = mean_h
            z_rhythm = mean_r
        kl_loss_h = kl_loss_r = 0.0
        if ret_kl:
            kl_loss_h = -0.5 * (1 + torch.log(var_h) - mean_h**2 - var_h).mean()
            kl_loss_r = -0.5 * (1 + torch.log(var_r) - mean_r**2 - var_r).mean()
        return z_rhythm, z_harmony, kl_loss_h + kl_loss_r

    @torch.jit.ignore
    def forward(self, x: torch.Tensor, shift=0):
        z_style, z_pitch, kl_loss = self._encode(x, rsample=True)
        if not self.no_shift:
            z_pitch = torch.roll(z_pitch, shifts=shift, dims=1)
        z_cat = torch.cat([z_pitch, z_style], dim=1)
        y = self.decoder(z_cat)
        return y, kl_loss

    def forward_mix(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight_harmony: torch.Tensor,
        weight_rhythm: torch.Tensor,
    ):
        z1_rhythm_mean, z1_harmony_mean, _ = self._encode(x1, rsample=False)
        z2_rhythm_mean, z2_harmony_mean, _ = self._encode(x2, rsample=False)
        z_style_mix = linear_interpolate(z1_rhythm_mean, z2_rhythm_mean, weight_rhythm)
        z_pitch_mix = linear_interpolate(
            z1_harmony_mean, z2_harmony_mean, weight_harmony
        )
        z_mix_cat = torch.cat([z_pitch_mix, z_style_mix], dim=1)
        ymix = self.decoder(z_mix_cat)
        return ymix
