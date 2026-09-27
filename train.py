"""Train the face embedding model on data/<person>/*.jpg (2+ people needed).

Uses triplet loss: pulls (anchor, positive) embeddings from the same
person together, pushes (anchor, negative) from a different person apart.
Reports triplet accuracy each epoch (fraction where the model correctly
ranks the positive closer than the negative) instead of classification
accuracy, since there's no fixed set of classes here.

After training, saves face_embeddings.pt — one averaged reference
embedding per enrolled person, computed from their own photos. That file
is what identify.py compares new faces against at inference time.
"""

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from dataset import TripletFaceDataset, load_face_image
from model import FaceEmbeddingCNN


def build_gallery(model, people):
    """One averaged, L2-normalized embedding per enrolled person."""
    model.eval()
    gallery = {}
    with torch.no_grad():
        for name, paths in people.items():
            imgs = torch.stack([load_face_image(p) for p in paths])
            embeddings = model(imgs)
            mean = embeddings.mean(dim=0)
            gallery[name] = F.normalize(mean, p=2, dim=0)
    return gallery


def main(epochs=20, batch_size=8, data_root="data", triplets_per_epoch=200, seed=0):
    torch.manual_seed(seed)
    dataset = TripletFaceDataset(data_root, triplets_per_epoch=triplets_per_epoch, seed=seed)
    if len(dataset.names) < 2:
        raise SystemExit(
            f"Only found {len(dataset.names)} enrolled people under '{data_root}/'. "
            "Need at least 2 (each with 2+ photos) — run enroll.py for each person first."
        )
    print(f"Training on {len(dataset.names)} enrolled people: {dataset.names}")

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FaceEmbeddingCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.TripletMarginLoss(margin=0.3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for anchor, positive, negative in train_loader:
            optimizer.zero_grad()
            a, p, n = model(anchor), model(positive), model(negative)
            loss = criterion(a, p, n)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * anchor.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                a, p, n = model(anchor), model(positive), model(negative)
                d_pos = (a - p).pow(2).sum(dim=1)
                d_neg = (a - n).pow(2).sum(dim=1)
                correct += (d_pos < d_neg).sum().item()
                total += anchor.size(0)
        triplet_acc = correct / total if total else float("nan")
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"— train loss: {running_loss / train_size:.4f} "
            f"— triplet accuracy: {triplet_acc:.2%} ({total} val triplets)"
        )

    torch.save(model.state_dict(), "face_embedding_cnn.pt")
    gallery = build_gallery(model, dataset.people)
    torch.save(gallery, "face_embeddings.pt")
    print(f"Saved model weights and an enrollment gallery for: {list(gallery.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--triplets-per-epoch", type=int, default=200)
    args = parser.parse_args()
    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_root=args.data_root,
        triplets_per_epoch=args.triplets_per_epoch,
    )
