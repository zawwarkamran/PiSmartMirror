"""Train the face-identity CNN on data/me/ and data/not_me/.

Fixes the original implement1.py training loop, which used
CrossEntropyLoss with the *input image itself* as the label — that
never converges to anything meaningful. This version does real binary
classification (BCEWithLogitsLoss) with a proper train/validation split
and reports validation accuracy every epoch.
"""

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from dataset import FaceDataset
from model import FaceIdentityCNN


def main(epochs=15, batch_size=8, data_root="data", seed=0):
    torch.manual_seed(seed)
    dataset = FaceDataset(data_root)
    if len(dataset) < 4:
        raise SystemExit(
            f"Only found {len(dataset)} images under '{data_root}/me' and "
            f"'{data_root}/not_me'. Run capture.py for both labels first "
            "(see README)."
        )

    val_size = max(1, int(0.2 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = FaceIdentityCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else float("nan")
        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"— train loss: {running_loss / train_size:.4f} "
            f"— val accuracy: {val_acc:.2%} ({total} val samples)"
        )

    torch.save(model.state_dict(), "face_identity_cnn.pt")
    print("Saved trained weights to face_identity_cnn.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    main(epochs=args.epochs, batch_size=args.batch_size, data_root=args.data_root)
