"""Enroll a new person: capture labeled, face-cropped photos from a webcam.

Any number of people can be enrolled this way — each gets their own
data/<name>/ folder. Multiple people are what train.py needs to learn a
useful embedding space (see README for why one person alone isn't enough).

Usage:
    python enroll.py zawwar --samples 40
    python enroll.py alex --samples 40
"""

import argparse
import os

import cv2

from model import IMG_SIZE


def enroll(name, samples=30, out_dir="data", camera=0):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera — is one connected/available?")

    save_dir = os.path.join(out_dir, name)
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
            path = os.path.join(save_dir, f"{name}_{existing + count}.jpg")
            cv2.imwrite(path, face_crop)
            count += 1
            print(f"Saved {path} ({count}/{samples})")
            break  # one face per frame is enough

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", help="Person's name — becomes their data/<name>/ folder")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    enroll(args.name, samples=args.samples, camera=args.camera)
