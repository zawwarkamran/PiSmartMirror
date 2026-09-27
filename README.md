# PiSmartMirror

A side project from university: I wanted a mirror that could tell whether
it was looking at me. Built for a Raspberry Pi with a webcam behind the
glass.

## How it works

```
webcam frame -> Haar cascade (find + crop the face) -> 64x64 crop
             -> embedding CNN (mine, PyTorch) -> 128-d vector
             -> nearest-neighbor match against enrolled people
```

OpenCV's Haar cascade handles finding a face in the frame — no need to
reinvent that part. The interesting part is how identification works.

The first version of this was a binary classifier: "is this me, yes/no."
That doesn't scale — a real multi-person system would need a brand new
classifier retrained from scratch for every new person added. So this
version trains an **embedding model** instead, with triplet loss: pull
same-person photos together in vector space, push different-person photos
apart. Enrolling someone new is then just: take a few photos, compute
their average embedding, store it — no retraining. Recognizing someone is
a nearest-neighbor lookup against everyone enrolled so far. This is the
same basic approach behind things like Face ID and Google Photos' face
grouping (FaceNet, ArcFace).

`model.py` is a small hand-written CNN — three conv/batchnorm/relu/maxpool
blocks feeding a linear head, L2-normalized to a 128-d embedding. No
pretrained backbone.

## Usage

```bash
pip install -r requirements.txt

# enroll at least 2 people (triplet loss needs a contrast to learn from)
python enroll.py zawwar --samples 40
python enroll.py alex --samples 40

# train the embedding model + build the enrollment gallery
python train.py --epochs 20

# live demo — opens a window that names each face it sees, or "unknown"
python identify.py
```

Adding person #3 later: `enroll.py`, then re-run `train.py`. The
embedding model itself only needs retraining occasionally as more people
join — the per-person gallery lookup is what actually scales for free.

## Data

No `data/` folder is committed here on purpose — I'm not putting face
photos of myself or anyone else into a public repo. `enroll.py` writes to
`data/<name>/` locally; run it yourself to build a training set before
running `train.py`.

Verified the full pipeline — triplet training, gallery construction, and
nearest-neighbor identification — end to end with placeholder images
before pushing this, so the code itself is solid. Bring your own photos
(and at least one other willing person) to get a real result.
