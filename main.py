import cv2
import numpy as np

captureWebcam = cv2.VideoCapture(0)

while True:
    success, video = captureWebcam.read()

    img = cv2.cvtColor(video, cv2.COLOR_BGR2HSV)

    # Lower and upper range of red color in HSV
    lowerRange1 = np.array([0, 50, 50])
    upperRange1 = np.array([10, 255, 255])
    lowerRange2 = np.array([170, 50, 50])
    upperRange2 = np.array([180, 255, 255])

    # Finding mask for red color
    mask1 = cv2.inRange(img, lowerRange1, upperRange1)
    mask2 = cv2.inRange(img, lowerRange2, upperRange2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Finding position of all contours
    if len(mask_contours) != 0:
        for mask_contour in mask_contours:
            if cv2.contourArea(mask_contour) > 500:
                x, y, w, h = cv2.boundingRect(mask_contour)
                cv2.rectangle(video, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow("mask image", mask)

    cv2.imshow("window", video)

    cv2.waitKey(1)