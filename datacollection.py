import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# -----------------------
# Camera
# -----------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

folder = r"E:\project\Data\yes"
os.makedirs(folder, exist_ok=True)

# Connections between hand landmarks
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

while True:

    success, img = cap.read()

    if not success:
        break

    imgOutput = img.copy()

    hands, img = detector.findHands(img, draw=False)

    if hands:

        hand = hands[0]
        x, y, w, h = hand["bbox"]

        # -------- Draw Hand Skeleton --------
        lmList = hand["lmList"]

        # Draw white lines
        for c in connections:
            x1, y1 = lmList[c[0]][0], lmList[c[0]][1]
            x2, y2 = lmList[c[1]][0], lmList[c[1]][1]
            cv2.line(imgOutput, (x1, y1), (x2, y2), (255,255,255), 2)

        # Draw red landmarks
        for lm in lmList:
            cv2.circle(imgOutput, (lm[0], lm[1]), 5, (0,0,255), cv2.FILLED)

        # Bounding box
        x1 = max(0, x-offset)
        y1 = max(0, y-offset)
        x2 = min(img.shape[1], x+w+offset)
        y2 = min(img.shape[0], y+h+offset)

        cv2.rectangle(imgOutput,(x1,y1),(x2,y2),(255,0,255),2)

        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:

            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

            cropH, cropW = imgCrop.shape[:2]
            aspectRatio = cropH/cropW

            if aspectRatio > 1:

                k = imgSize/cropH
                wCal = math.ceil(cropW*k)

                imgResize = cv2.resize(imgCrop,(wCal,imgSize))
                wGap = math.ceil((imgSize-wCal)/2)

                imgWhite[:,wGap:wGap+wCal] = imgResize

            else:

                k = imgSize/cropW
                hCal = math.ceil(cropH*k)

                imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                hGap = math.ceil((imgSize-hCal)/2)

                imgWhite[hGap:hGap+hCal,:] = imgResize

            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if hands and imgCrop.size != 0:
            counter += 1
            filename = os.path.join(folder, f"Image_{time.time()}.jpg")
            cv2.imwrite(filename, imgWhite)
            print("Saved:", counter)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()