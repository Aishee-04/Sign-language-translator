import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Cannot open camera!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ---------------- Hand Detector ----------------
detector = HandDetector(maxHands=1)

# ---------------- Classifier ----------------
classifier = Classifier(
    r"E:\Downloads\converted_keras\keras_model.h5",
    r"E:\Downloads\converted_keras\labels.txt"
)

# Read labels automatically from labels.txt
with open(r"E:\Downloads\converted_keras\labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] if " " in line else line.strip()
              for line in f.readlines()]

offset = 20
imgSize = 300

while True:

    success, img = cap.read()

    if not success:
        print("Failed to grab frame")
        continue

    # Mirror image
    img = cv2.flip(img, 1)

    # Keep a copy
    imgOutput = img.copy()

    # Detect hand
    hands, img = detector.findHands(img)

    if hands:

        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # Safe crop
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size > 0:

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            cropH, cropW = imgCrop.shape[:2]

            aspectRatio = cropH / cropW

            if aspectRatio > 1:

                k = imgSize / cropH
                wCal = math.ceil(cropW * k)

                imgResize = cv2.resize(imgCrop, (wCal, imgSize))

                wGap = (imgSize - wCal) // 2

                imgWhite[:, wGap:wGap + wCal] = imgResize

            else:

                k = imgSize / cropW
                hCal = math.ceil(cropH * k)

                imgResize = cv2.resize(imgCrop, (imgSize, hCal))

                hGap = (imgSize - hCal) // 2

                imgWhite[hGap:hGap + hCal, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            confidence = prediction[index] * 100

            # Draw box
            cv2.rectangle(
                imgOutput,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                3
            )

            # Draw label
            cv2.rectangle(
                imgOutput,
                (x1, y1 - 60),
                (x1 + 250, y1),
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.putText(
                imgOutput,
                f"{labels[index]} {confidence:.1f}%",
                (x1 + 10, y1 - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    # Show live camera
    cv2.imshow("Prediction", imgOutput)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()