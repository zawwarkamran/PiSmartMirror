"""Capture labeled, face-cropped training samples from a webcam.

Replaces the original TakeImg() from implement1.py, which saved whole
resized frames (background included) with no face localization. This
version uses OpenCV's Haar cascade to find the face first, crops just
that region, and saves it into data/<label>/ for dataset.py to load.

Usage:
    python capture.py me --samples 40
    python capture.py not_me --samples 40
"""

import argparse
import os

import cv2

from model import IMG_SIZE


def capture(label, samples=30, out_dir="data", camera=0):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera — is one connected/available?")

    save_dir = os.path.join(out_dir, label)
    os.makedirs(save_dir, exist_ok=True)
    existing = len([f for f in os.listdir(save_dir) if f.endswith(".jpg")])

    count = 0
    while count < samples:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from camera")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            path = os.path.join(save_dir, f"{label}_{existing + count}.jpg")
            cv2.imwrite(path, face_crop)
            count += 1
            print(f"Saved {path} ({count}/{samples})")
            break  # one face per frame is enough

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=["me", "not_me"], help="Class for these samples")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    capture(args.label, samples=args.samples, camera=args.camera)
