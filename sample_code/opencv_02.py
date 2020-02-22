# import cv2
#
# image = cv2.imread('test.jpg')
# print(image.shape)
# print(image.size)
#
# px = image[100,100]
# print(px)
# print(px[2])/

# import cv2
# import matplotlib.pyplot as plt
# import time
#
# image = cv2.imread('test.jpg')
#
# image[0:100, 0:100] = [0,0,0]
#
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()
#
# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('test.jpg')
# roi=image[200:350,  50:200]
# image[0:150, 0:150] = roi
#
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()

import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../rsc/test.jpg')
image[:,:,0]=0
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()