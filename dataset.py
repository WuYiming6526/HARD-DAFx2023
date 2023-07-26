import torch
from torch.utils.data import Dataset
from librosa.util import find_files
import torchaudio
from torchaudio.functional import resample
import numpy as np


def kaiser_resample(x, orig_sr, targ_sr):
    return resample(
        x,
        orig_sr,
        targ_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="kaiser_window",
        beta=14.769656459379492,
    )


class AudioDataset(Dataset):
    def __init__(
        self,
        path,
        n_files: int = -1,
        sample_rate: int = 22050,
        audio_channels=1,
        batch_audio_length: int = 44100,
        batch_start_pos=-1,
        audio_ext=None,
        **kwargs
    ):
        self.sample_rate = sample_rate
        self.list_audiofile = []
        self.list_audioinfo = []
        if isinstance(path, list):
            flist = sum([find_files(p, ext=audio_ext) for p in path], start=[])
        else:
            assert isinstance(path, str)
            flist = find_files(path, ext=audio_ext)
        if n_files > 0:
            randidx = np.random.permutation(len(flist))[:n_files]
            flist = np.array(flist)[randidx]
        for f in flist:
            audio_info = torchaudio.info(f)
            if audio_info.num_frames < batch_audio_length:
                continue
            self.list_audiofile.append(f)
            self.list_audioinfo.append(audio_info)

        self.audio_channels = audio_channels
        self.batch_audio_length = batch_audio_length
        self.batch_start_pos = batch_start_pos
        self.list_filename = [p.split("/")[-1] for p in self.list_audiofile]

    def __len__(self):
        return len(self.list_audiofile)

    def __getitem__(self, index):
        f = self.list_audiofile[index]
        info = self.list_audioinfo[index]
        batch_length = self.batch_audio_length
        wav, sr = torchaudio.load(f)
        if self.batch_start_pos < 0:
            audio_len = wav.shape[-1]
            pos_start = np.random.randint(audio_len - batch_length)
        else:
            pos_start = self.batch_start_pos * info.sample_rate

        wav = wav[:, pos_start : pos_start + batch_length]
        if self.audio_channels == 1 and wav.shape[0] == 2:
            wav = wav.mean(0, keepdim=True)
        if self.audio_channels == 2 and wav.shape[0] == 1:
            wav = torch.cat([wav, wav], dim=0)
        if sr != self.sample_rate:
            wav = kaiser_resample(wav, sr, self.sample_rate)

        assert wav.shape[0] == self.audio_channels
        wav = torch.clamp(wav, min=-1.0, max=1.0)

        return wav

    def get_full_audio(self, index):
        buffer = self.list_audiodata[index]
        info = self.list_audioinfo[index]
        wav, sr = torchaudio.load(buffer, format=info.encoding)
        if sr != self.sample_rate:
            wav = resample(
                wav,
                sr,
                self.sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        if self.audio_channels == 1 and wav.shape[0] == 2:
            wav = wav.mean(0, keepdim=True)
        if self.audio_channels == 2 and wav.shape[0] == 1:
            wav = torch.cat([wav, wav], dim=0)
        return wav
