import matplotlib.pyplot as plt
import lightning.pytorch as pl
from models import vaegan
import torch
import torch.optim
import torchaudio
import os
from torch.utils.data import DataLoader
import dataset
import numpy as np
from models import spectrogram, vaegan

from pedalboard import Pedalboard, PitchShift
from joblib import Parallel, delayed


class LitModuleBase(pl.LightningModule):
    def __init__(
        self,
        sample_rate: int,
        dset_config: dict = None,
        num_workers: int = 1,
        batch_size: int = 32,
        batch_length_factor: int = 1024,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.dset_config = dset_config
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.batch_length_factor = batch_length_factor

    def train_dataloader(self):
        train_set = dataset.AudioDataset(**self.dset_config)
        return DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        val_config = self.dset_config.copy()
        val_config["n_files"] = 8
        val_set = dataset.AudioDataset(**val_config)
        return DataLoader(
            val_set,
            batch_size=2,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
        )

    def validation_step(self, batch, batch_idx):
        wav = batch
        x = self.melspec.forward(wav)
        y, _ = self.model.forward(x)
        # with torch.enable_grad():
        #    y_invert_cpu = self.inverse_melspec(y).cpu()
        wav_cpu = wav.cpu().float()
        x_np = x.cpu().flatten(1, 2).numpy()
        y_np = y.cpu().flatten(1, 2).numpy()
        logdir = self.trainer.log_dir
        plt.imsave(os.path.join(logdir, "melspec_x1.png"), x_np[0], origin="lower")
        plt.imsave(os.path.join(logdir, "melspec_y1.png"), y_np[0], origin="lower")
        plt.imsave(os.path.join(logdir, "melspec_x2.png"), x_np[1], origin="lower")
        plt.imsave(os.path.join(logdir, "melspec_y2.png"), y_np[1], origin="lower")
        torchaudio.save(
            os.path.join(logdir, "audio_x1.wav"), wav_cpu[0], self.sample_rate
        )
        torchaudio.save(
            os.path.join(logdir, "audio_x2.wav"), wav_cpu[1], self.sample_rate
        )

        y_mix_1 = self.model.forward_mix(x[0:1, ...], x[1:2, ...], 0.0, 1.0)
        y_mix_2 = self.model.forward_mix(x[0:1, ...], x[1:2, ...], 1.0, 0.0)
        y_mix_1_np = y_mix_1.cpu().flatten(1, 2).numpy()
        y_mix_2_np = y_mix_2.cpu().flatten(1, 2).numpy()
        plt.imsave(
            os.path.join(logdir, "melspec_mix_1.png"), y_mix_1_np[0], origin="lower"
        )
        plt.imsave(
            os.path.join(logdir, "melspec_mix_2.png"), y_mix_2_np[0], origin="lower"
        )


class LitMelspecDisentangleVAE(LitModuleBase):
    def __init__(
        self,
        vae_name: str,
        vae_args: dict,
        sample_rate: int = 22050,
        n_mels: int = 256,
        mono_mel: bool = False,
        hop_size: int = 256,
        beta: float = 1.0,
        clip_gradient: float = 0.0,
        lr: float = 1e-4,
        lr_decay: float = 1.0,
        dset_config: dict = None,
        num_workers: int = 1,
        batch_size: int = 32,
        batch_length_factor: int = 1024,
        pitch_shift_mode: str = "rubberband",
    ):
        super().__init__(
            sample_rate, dset_config, num_workers, batch_size, batch_length_factor
        )
        self.save_hyperparameters()
        vae_class = {
            "PitchDisentangleVAE": vaegan.PitchDisentangleVAEGAN,
        }[vae_name]
        self.model = vae_class(**vae_args)
        self.melspec = spectrogram.MelSpecEncoder(
            n_mels=n_mels,
            hop_size=hop_size,
            sample_rate=sample_rate,
            mono=mono_mel,
            supress_channel=False,
        )
        self.beta = beta
        self.lr = lr
        self.lr_decay = lr_decay
        self.clip_gradient = clip_gradient
        self.pitch_shift_mode = pitch_shift_mode

    def forward(self, x: torch.Tensor):
        x = self.melspec.forward(x)
        y, _ = self.model.forward(x)
        return y

    def forward_mix(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        weight_harmony: torch.Tensor,
        weight_rhythm: torch.Tensor,
    ):
        s1, s2 = self.melspec(x1), self.melspec(x2)
        y = self.model.forward_mix(s1, s2, weight_harmony, weight_rhythm)
        return y

    def training_step(self, batch, batch_idx):
        x, x_shift, n_shift = batch
        log = {}
        x = self.melspec.forward(x)
        x_shift = self.melspec.forward(x_shift)
        y, loss_kl = self.model.forward(x_shift, -n_shift)
        loss_rec = self.model.rec_loss(x, y)

        log["train/loss_rec"] = loss_rec
        log["train/loss_kl"] = loss_kl
        self.log_dict(log)

        loss = loss_kl * self.beta + loss_rec
        return loss

    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        def pitch_shift_process(y, sr, n_shift):
            if n_shift == 0:
                return y
            board = Pedalboard([PitchShift(n_shift)])
            y_shift = board(y, sr, reset=True)
            return y_shift

        if self.trainer.training:
            batch_np = batch.numpy()
            batch_size = batch_np.shape[0]
            n_shift = np.random.randint(-7, 8)
            if self.pitch_shift_mode == "rubberband":
                batch_shift = Parallel(n_jobs=batch_size)(
                    delayed(pitch_shift_process)(
                        batch_np[i, ...], self.sample_rate, n_shift
                    )
                    for i in range(batch_size)
                )
                min_len = min([b.shape[-1] for b in batch_shift])
                factor = self.batch_length_factor
                min_len = min_len // factor * factor
                batch_shift = [b[..., :min_len] for b in batch_shift]
                batch_shift = torch.from_numpy(np.stack(batch_shift, axis=0))
                batch = batch[..., :min_len]
                assert batch.shape[-1] == batch_shift.shape[-1]
            else:
                batch_shift = batch.clone()
                """
                batch_shift = torchaudio.functional.pitch_shift(
                    waveform=batch,
                    sample_rate=self.sample_rate,
                    n_steps=n_shift,
                    n_fft=4096,
                    hop_length=1024
                )
                """

            return batch, batch_shift, n_shift
        else:
            return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if self.trainer.training:
            wav, wav_shift, n_shift = batch
            return wav.to(device), wav_shift.to(device), n_shift
        else:
            return batch.to(device)

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        if self.trainer.training:
            wav, wav_shift, n_shift = batch
            if self.pitch_shift_mode == "torchaudio":
                torchaudio.functional.pitch_shift(
                    waveform=wav_shift,
                    sample_rate=self.sample_rate,
                    n_steps=n_shift,
                    n_fft=4096,
                    hop_length=1024,
                )
            return wav, wav_shift, n_shift
        else:
            return batch

    def on_training_epoch_end(self, outputs):
        if self.lr_decay < 1.0:
            sch_adv, sch_dis = self.lr_schedulers()
            sch_adv.step()
            sch_dis.step()

    def configure_optimizers(self):
        lr = self.lr
        adv_params = list(self.model.encoder.parameters()) + list(
            self.model.decoder.parameters()
        )
        opt_adv = torch.optim.RAdam(adv_params, lr=lr, betas=(0.8, 0.99))
        opt_discrim = torch.optim.RAdam(
            self.model.discriminator.parameters(), lr=lr, betas=(0.8, 0.99)
        )
        if self.lr_decay >= 1.0:
            return opt_adv, opt_discrim

        else:
            sch_adv = torch.optim.lr_scheduler.ExponentialLR(
                opt_adv, self.lr_decay, verbose=True
            )
            sch_dis = torch.optim.lr_scheduler.ExponentialLR(
                opt_discrim, self.lr_decay, verbose=True
            )
            schedulers = [sch_adv, sch_dis]

            return [opt_adv, opt_discrim], schedulers


class LitMelspecDisentangleVAEGAN(LitMelspecDisentangleVAE):
    def __init__(
        self,
        vae_name: str,
        vae_args: dict,
        sample_rate: int = 22050,
        n_mels: int = 256,
        mono_mel: bool = False,
        hop_size: int = 256,
        beta: float = 1.0,
        clip_gradient: float = 0.0,
        rec_loss_weight: float = 1.0,
        adv_loss_weight: float = 1.0,
        feat_loss_weight: float = 1.0,
        lr: float = 1e-4,
        lr_decay: float = 1.0,
        dset_config: dict = None,
        num_workers: int = 1,
        batch_size: int = 32,
        batch_length_factor: int = 1024,
        pitch_shift_mode: str = "rubberband",
    ):
        super().__init__(
            vae_name,
            vae_args,
            sample_rate,
            n_mels,
            mono_mel,
            hop_size,
            beta,
            clip_gradient,
            lr,
            lr_decay,
            dset_config,
            num_workers,
            batch_size,
            batch_length_factor,
            pitch_shift_mode,
        )
        self.rec_loss_weight = rec_loss_weight
        self.adv_loss_weight = adv_loss_weight
        self.feat_loss_weight = feat_loss_weight
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, x_shift, n_shift = batch
        log = {}
        opt_adv, opt_dis = self.optimizers()

        x = self.melspec.forward(x)
        x_shift = self.melspec.forward(x_shift)

        y, loss_kl = self.model.forward(x_shift, -n_shift)
        for _ in range(1):
            loss_disc = self.model.discriminator_loss(y, x)
            opt_dis.zero_grad()
            self.manual_backward(loss_disc)
            opt_dis.step()
            log["train/loss_dis"] = loss_disc
        loss_adv, loss_feat = self.model.adversarial_generative_loss(x, y)
        loss_rec = self.model.rec_loss(x, y)

        opt_adv.zero_grad()
        self.manual_backward(
            (loss_rec + loss_kl * self.beta) * self.rec_loss_weight
            + (loss_adv * self.adv_loss_weight)
            + (loss_feat * self.feat_loss_weight)
        )
        opt_adv.step()

        log["train/loss_rec"] = loss_rec
        log["train/loss_kl"] = loss_kl
        log["train/loss_adv"] = loss_adv
        log["train/loss_feat"] = loss_feat
        self.log_dict(log)
