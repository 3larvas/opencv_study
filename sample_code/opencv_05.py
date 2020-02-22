# 임계점 처리

# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE)
#
# images = []
# ret, thres1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
# ret, thres2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
# ret, thres3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
# ret, thres4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
# ret, thres5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
# images.append(thres1)
# images.append(thres2)
# images.append(thres3)
# images.append(thres4)
# images.append(thres5)
#
# for i in images:
#   plt.imshow(cv2.cvtColor(i, cv2.COLOR_GRAY2RGB))
#   plt.show()


import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../rsc/test.jpg', cv2.IMREAD_GRAYSCALE)

ret, thres1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
thres2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 3)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
plt.show()

plt.imshow(cv2.cvtColor(thres1, cv2.COLOR_GRAY2RGB))
plt.show()

plt.imshow(cv2.cvtColor(thres2, cv2.COLOR_GRAY2RGB))
plt.show()
#
# import cv2
# import numpy as np
#
#
# def change_color(x):
#   r = cv2.getTrackbarPos("R", "Image")
#   g = cv2.getTrackbarPos("G", "Image")
#   b = cv2.getTrackbarPos("B", "Image")
#   image[:] = [b, g, r]
#   cv2.imshow('Image', image)
#
#
# image = np.zeros((600, 400, 3), np.uint8)
# cv2.namedWindow("Image")
#
# cv2.createTrackbar("R", "Image", 0, 255, change_color)
# cv2.createTrackbar("G", "Image", 0, 255, change_color)
# cv2.createTrackbar("B", "Image", 0, 255, change_color)
#
# cv2.imshow('Image', image)
# cv2.waitKey(0)