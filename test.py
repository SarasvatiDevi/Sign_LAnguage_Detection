import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open Camera.")
    exit()

detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\TGM\Downloads\AI_MODEL\keras_model.h5",
                        "C:\\Users\TGM\Downloads\AI_MODEL\labels.txt")

offset = 20
imgSize = 224
labels = [
    "Bye", "Call", "Good Luck", "Hello", "I Love You", "Left", "Loser", "Nice", "No", "Not okay",
    "Ok", "Peace", "Please", "Power", "Right", "Rock", "Smile", "Stop", "Thank you", "Yes"
]

try:
    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture image.")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

                imgWhite = cv2.resize(imgWhite, (224, 224))

                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                cv2.rectangle(imgOutput, (x - offset, y - offset - 70),
                              (x - offset + 400, y - offset + 60 - 50),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(imgOutput, labels[index], (x, y - 30),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (0, 255, 0), 4)

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)

            except Exception as e:
                print(f"Error during processing: {e}")

        cv2.imshow('Image', imgOutput)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Exiting on user request.")
            break

except KeyboardInterrupt:
    print("Program interrupted. Exiting...")

finally:
    cap.release()
    cv2.destroyAllWindows()
