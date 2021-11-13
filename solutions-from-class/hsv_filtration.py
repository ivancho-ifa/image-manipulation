import numpy as np
import matplotlib.pyplot as plt
import cv2


def onChangeCallback(x):
    pass

# setup windows positions
cv2.namedWindow('input')
cv2.moveWindow('input', 0, 20)

cv2.namedWindow('trackbars')
cv2.moveWindow('trackbars', 1000, 20)

cv2.namedWindow('mask')
cv2.moveWindow('mask', 500, 20)

cv2.namedWindow('output')
cv2.moveWindow('output', 500, 20)

# setup trackbars
cv2.createTrackbar('min H', 'trackbars', 0, 255, onChangeCallback)
cv2.createTrackbar('max H', 'trackbars', 0, 255, onChangeCallback)
cv2.createTrackbar('min S', 'trackbars', 0, 255, onChangeCallback)
cv2.createTrackbar('max S', 'trackbars', 0, 255, onChangeCallback)
cv2.createTrackbar('min V', 'trackbars', 0, 255, onChangeCallback)
cv2.createTrackbar('max V', 'trackbars', 0, 255, onChangeCallback)

# read input image
img = cv2.imread('color.jpg', cv2.IMREAD_COLOR)

# convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while True:
    hMin = cv2.getTrackbarPos('min H', 'trackbars')
    hMax = cv2.getTrackbarPos('max H', 'trackbars')
    sMin = cv2.getTrackbarPos('min S', 'trackbars')
    sMax = cv2.getTrackbarPos('max S', 'trackbars')
    vMin = cv2.getTrackbarPos('min V', 'trackbars')
    vMax = cv2.getTrackbarPos('max V', 'trackbars')

    # define range of color in HSV
    lowerBoundaries = np.array([hMin, sMin, vMin])
    upperBoundaries = np.array([hMax, sMax, vMax])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lowerBoundaries, upperBoundaries)

    # Bitwise-AND mask and original image
    output = cv2.bitwise_and(img, img, mask = mask)

    cv2.imshow('input', img)
    cv2.imshow('mask', mask)
    cv2.imshow('output', output)

    # close on escape key
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()