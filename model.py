import torch.nn.functional as F
import torch.nn as nn

IMG_SIZE = 64
EMBEDDING_DIM = 128


class FaceEmbeddingCNN(nn.Module):
    """Maps a 64x64 RGB face crop to a 128-d, L2-normalized embedding.

    Not a per-person classifier — this is trained so that embeddings from
    the same person land close together in space, and different people
    land far apart (triplet loss, see train.py). Recognizing a new person
    at inference time is a nearest-neighbor lookup against enrolled
    embeddings (identify.py), not a retrained classifier, so adding a new
    user just means storing one more vector.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 -> 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, EMBEDDING_DIM),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.embed(x)
        return F.normalize(x, p=2, dim=1)
