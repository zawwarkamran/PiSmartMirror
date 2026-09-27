# PiProjet — Smart Mirror Face ID

A 2020 side project: a Raspberry Pi "smart mirror" that recognizes whether the
face it sees is mine. Rebuilt in 2026 to fix bugs in the original and turn it
into a working, correct pipeline.

## What changed from the original

The original 2020 version had two disconnected pieces and a training bug:

- `tester.py` did real-time face **detection** with OpenCV's Haar cascade —
  this part worked, but only located faces, it didn't identify anyone.
- `implement1.py` / `torchdetector.py` defined a custom PyTorch CNN meant to
  **identify** faces, but the training loop used each image as its own label
  with `CrossEntropyLoss` — a setup that doesn't correspond to any real
  classification task, so it never learned anything meaningful.

This version fixes that: the Haar cascade now handles **localization**
(finding and cropping the face), and the CNN handles **identification**
(a proper binary classifier — "is this face mine?" — trained with
`BCEWithLogitsLoss` and a real train/validation split).

## Architecture

```
webcam frame -> Haar cascade (locate face) -> crop to 64x64
             -> FaceIdentityCNN (3 conv blocks + linear head) -> sigmoid -> me / not me
```

`FaceIdentityCNN` (`model.py`) is a small hand-written CNN: three
conv/batchnorm/relu/maxpool blocks (3→32→64→128 channels) followed by a
linear classification head — no pretrained backbone, no library model.

## Usage

```bash
pip install -r requirements.txt

# collect labeled face crops (repeat for both classes)
python capture.py me --samples 40
python capture.py not_me --samples 40

# train
python train.py --epochs 15

# live demo (opens a window with a labeled bounding box)
python detect_live.py
```

## A note on the data

This repo does **not** include a `data/` folder of face photos, and none
should be added — publishing identifiable photos of yourself or anyone else
into a public repo isn't something to do casually. `capture.py` writes to
`data/me/` and `data/not_me/` locally (both are gitignored); run it yourself
to generate your own training set before running `train.py`.

The code has been verified to run correctly end-to-end (checked with
synthetic placeholder images to confirm the training loop, loss, and
validation accuracy all compute correctly) — the architecture and pipeline
are sound, but no model is shipped pretrained. Run it with your own data to
get a real result.

## Scope, honestly

This is a personal-scale project, not production computer vision — no
keypoint/landmark detection, no large dataset, no deployment. What it does
demonstrate: designing a CNN architecture from scratch in PyTorch, correctly
framing and training a classification task, and combining a classical CV
technique (Haar cascades) with a learned model in one pipeline.
