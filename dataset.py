import os

import cv2
import torch
from torch.utils.data import Dataset

from model import IMG_SIZE


class FaceDataset(Dataset):
    """Loads labeled face crops from data/me/*.jpg and data/not_me/*.jpg.

    Label 1 = me, label 0 = not_me. Both folders are populated by capture.py
    and gitignored — see README for why.
    """

    def __init__(self, root="data"):
        self.samples = []
        for label, folder in enumerate(["not_me", "me"]):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder_path, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor, torch.tensor([label], dtype=torch.float32)
