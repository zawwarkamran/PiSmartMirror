"""Live smart-mirror demo: Haar cascade locates faces in the webcam feed,
the trained CNN (face_identity_cnn.pt) classifies each one as "me" or not.

Run train.py first to produce face_identity_cnn.pt.
"""

import cv2
import torch

from model import IMG_SIZE, FaceIdentityCNN


def main(weights="face_identity_cnn.pt", camera=0):
    model = FaceIdentityCNN()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

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
            face_crop = cv2.resize(face_crop, (IMG_SIZE, IMG_SIZE))
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0

            with torch.no_grad():
                prob = torch.sigmoid(model(tensor)).item()

            is_me = prob > 0.5
            label = f"me ({prob:.0%})" if is_me else f"not me ({1 - prob:.0%})"
            color = (0, 255, 0) if is_me else (0, 0, 255)
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
