import torch
from torch.utils.data import Dataset
import os
import torchaudio
from torchvision.io import read_video
import torchlibrosa

SAMPLE_RATE   = 16_000
N_FFT         = 1024
WIN_LENGTH    = None
# Stage 1 pre-training
HOP_LENGTH = 250
# Stage 2 LDM training
# HOP_LENGTH = 256
N_MELS        = 128
F_MIN, F_MAX  = 0, SAMPLE_RATE // 2
POWER         = 2.0           # power = 2 → power-spectrogram; 1 → amplitude

TARGET_FPS   = 4        # keep it light; original is 30 fps
TARGET_SIZE  = 224       # output H = W = 224
FIXED_NUM_FRAMES = 40

mel_transform = torchaudio.transforms.MelSpectrogram(
# mel_transform = torchlibrosa.stft.MelSpectrogram(
    sample_rate   = SAMPLE_RATE,
    n_fft         = N_FFT,
    win_length    = WIN_LENGTH,
    hop_length    = HOP_LENGTH,
    f_min         = F_MIN,
    f_max         = F_MAX,
    n_mels        = N_MELS,
    power         = POWER,
)
class VidSpectroDataset (Dataset):
    def __init__(self, data_path, device):
        super().__init__()
        self.device = device
        self.data_path = data_path
        self.ids = self.get_ids()

    def aud_to_spec(self, name):
        wav, sr = torchaudio.load(f"{self.data_path}/{name}.wav")  # (channels, time)

        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

        wav = wav.mean(dim=0, keepdim=True)  # mono

        if wav.shape[1] > SAMPLE_RATE:
            wav = wav[:, :SAMPLE_RATE]
        elif wav.shape[1] < SAMPLE_RATE:
            pad_len = SAMPLE_RATE - wav.shape[1]
            wav = torch.nn.functional.pad(wav, (0, pad_len))

        mel = mel_transform(wav)  # (1, n_mels, time)
        mel = torchaudio.functional.amplitude_to_DB(mel, multiplier=10.0, amin=1e-10, db_multiplier=0)
        return mel

    def gen_vid(self, name):
        # read_video returns (T, H, W, C) uint8
        frames, _, meta = read_video(f"{self.data_path}/{name}.mp4", pts_unit="sec")

        # 1) evenly temporally sample fixed frame size
        T = frames.shape[0]
        idxs = torch.linspace(0, T - 1, FIXED_NUM_FRAMES, dtype=torch.int)
        frames = frames[idxs]

        # 2) resize spatially and scale to [0, 1]
        frames = (
            torch.nn.functional.interpolate(
                frames.permute(0, 3, 1, 2).float(),       # (T, C, H, W)
                size=TARGET_SIZE,
                mode="bilinear",
                align_corners=False,
            ) / 255.0
        )                                                 # (T, C, H, W) float32

        # 3) reorder to (C, T, H, W)
        frames = frames.permute(1, 0, 2, 3).contiguous()  # (C, T, H, W)

        return frames

    def get_ids(self):
        seen = set()
        res = []
        for f in os.listdir(self.data_path):
            name, ext = os.path.splitext(f)
            if ext == ".wav" and name not in seen:
                res.append(name)
                seen.add(name)
        return res

    @staticmethod
    def collate_fn(batch):
        return {
            'audio': torch.stack([x['audio'] for x in batch]),
            'video': torch.stack([x['video'] for x in batch])
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        spec = self.aud_to_spec(name)
        vid = self.gen_vid(name)
        # print(f"idx {idx:4d} | id: {name} | vid: {vid.shape} | aud: {spec.shape}")
        return {
          "audio": spec,
          "video": vid
        }

        # in memory data
        # return self.data[idx][0], self.data[idx][1]
