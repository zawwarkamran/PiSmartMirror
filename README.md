# PiProjet — Smart Mirror Face ID

A side project from university: I wanted a mirror that could tell whether it
was looking at me. Built for a Raspberry Pi with a webcam behind the glass.

## How it works

```
webcam frame -> Haar cascade (find + crop the face) -> 64x64 crop
             -> small CNN (mine, PyTorch) -> sigmoid -> me / not me
```

OpenCV's Haar cascade handles finding a face in the frame — no need to
reinvent that part. The actual identification is a small CNN I wrote from
scratch: three conv/batchnorm/relu/maxpool blocks feeding into a linear
head, trained as a binary classifier.

The original 2020 training loop had a bug — the loss function I was using
didn't actually match what I was trying to do, so it wasn't learning
anything real. Went back and fixed it: proper binary classification loss,
a real train/validation split, and it now reports accuracy per epoch.

## Usage

```bash
pip install -r requirements.txt

# collect labeled face crops (repeat for both classes)
python capture.py me --samples 40
python capture.py not_me --samples 40

# train
python train.py --epochs 15

# live demo — opens a window with a labeled bounding box
python detect_live.py
```

## Data

No `data/` folder is committed here on purpose — I'm not putting face
photos of myself or anyone else into a public repo. `capture.py` writes to
`data/me/` and `data/not_me/` locally; run it yourself to build a training
set before running `train.py`.

Verified the pipeline runs correctly end-to-end with placeholder images
before pushing this, so the code itself is solid — bring your own photos
to get a real result.
