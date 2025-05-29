import torch
from torch.utils.data import Dataset
import os

class VidSpectroDataset (Dataset):
    def __init__(self, data_path, device):
        super().__init__()
        self.device = device
        self.data_path = data_path
        self.data = self.get_data()

    def get_data(self):
        i = 0
        seen = set()
        tmp = {}
        for f in sorted(os.listdir(self.data_path)):
            name, ext = os.path.splitext(f)
            if name not in seen:
                aud = torch.load(f"{self.data_path}/audio/{name}_audio.pt")
                vid = torch.load(f"{self.data_path}/video/{name}_video.pt")
                tmp[i] = (aud, vid)
                seen.add(name)
                i += 1
        return tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]