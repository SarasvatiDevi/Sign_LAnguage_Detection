import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Open the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam/video file.")
    exit()

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224
counter = 0
folder = "C:\\Users\\TGM\\PycharmProjects\\AI_Project\\Data\\No"

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            if 'bbox' in hand:
                x, y, w, h = hand['bbox']
                hImg, wImg, _ = img.shape
                y1, y2 = max(0, y-offset), min(hImg, y+h+offset)
                x1, x2 = max(0, x-offset), min(wImg, x+w+offset)
                imgCrop = img[y1:y2, x1:x2]
                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCropShape = imgCrop.shape
                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    if imgResize.shape[1] <= imgWhite.shape[1]:
                        imgWhite[:, wGap:wGap+imgResize.shape[1]] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    if imgResize.shape[0] <= imgWhite.shape[0]:
                        imgWhite[hGap:hGap+imgResize.shape[0], :] = imgResize

                cv2.imshow("ImageCrop", imgCrop)
                cv2.imshow("ImageWhite", imgWhite)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord("s"):
            counter += 1
            cv2.imwrite(f'{folder}/Image_{time.time()}.png', imgWhite)
            print(counter)

except KeyboardInterrupt:
    print("Program interrupted. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
