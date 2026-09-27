"""Live smart-mirror demo: Haar cascade locates faces in the webcam feed,
the trained embedding model + a nearest-neighbor lookup names each one.

Run train.py first to produce face_embedding_cnn.pt and face_embeddings.pt
(the enrollment gallery). Adding a new person later is just: enroll.py,
re-run train.py (or add their embedding to the gallery without retraining
the backbone, if it already generalizes well enough) — no per-person
classifier to maintain.
"""

import cv2
import torch
import torch.nn.functional as F

from model import IMG_SIZE, FaceEmbeddingCNN

MATCH_THRESHOLD = 0.6  # cosine similarity; tune based on your own enrollment data


def load_face_image_bgr(frame_bgr):
    face = cv2.resize(frame_bgr, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def identify(embedding, gallery):
    """Nearest enrolled person by cosine similarity, or None if no match clears the threshold."""
    best_name, best_score = None, -1.0
    for name, ref in gallery.items():
        score = F.cosine_similarity(embedding, ref.unsqueeze(0)).item()
        if score > best_score:
            best_name, best_score = name, score
    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score
    return None, best_score


def main(weights="face_embedding_cnn.pt", gallery_path="face_embeddings.pt", camera=0):
    model = FaceEmbeddingCNN()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    gallery = torch.load(gallery_path)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    cap = cv2.VideoCapture(camera)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            tensor = load_face_image_bgr(face_crop)

            with torch.no_grad():
                embedding = model(tensor)
            name, score = identify(embedding, gallery)

            label = f"{name} ({score:.2f})" if name else f"unknown ({score:.2f})"
            color = (0, 255, 0) if name else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )

        cv2.imshow("Smart Mirror - Face ID", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
