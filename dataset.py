import os
import random

import cv2
import torch
from torch.utils.data import Dataset

from model import IMG_SIZE


def load_face_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class TripletFaceDataset(Dataset):
    """Samples (anchor, positive, negative) triplets from data/<person>/*.jpg.

    Any number of people/folders works — that's the point of training on
    triplets instead of per-person classes: the model learns a general
    notion of "same person vs different person," not a fixed label set.
    Needs at least 2 people, each with at least 2 photos.
    """

    def __init__(self, root="data", triplets_per_epoch=200, seed=0):
        self.people = {}
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                folder = os.path.join(root, name)
                if not os.path.isdir(folder):
                    continue
                imgs = [
                    os.path.join(folder, f)
                    for f in sorted(os.listdir(folder))
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                if len(imgs) >= 2:
                    self.people[name] = imgs

        self.names = list(self.people.keys())
        self.triplets_per_epoch = triplets_per_epoch
        self.rng = random.Random(seed)

    def __len__(self):
        return self.triplets_per_epoch

    def __getitem__(self, idx):
        if len(self.names) < 2:
            raise RuntimeError(
                f"Need at least 2 enrolled people with 2+ photos each; "
                f"found {len(self.names)}. Run enroll.py for more people."
            )

        anchor_name = self.rng.choice(self.names)
        negative_name = self.rng.choice([n for n in self.names if n != anchor_name])

        anchor_path, positive_path = self.rng.sample(self.people[anchor_name], 2)
        negative_path = self.rng.choice(self.people[negative_name])

        return load_face_image(anchor_path), load_face_image(positive_path), load_face_image(negative_path)
