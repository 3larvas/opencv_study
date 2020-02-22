# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
#
# image = cv2.imread('test.jpg')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()
#
# size = 4
# kernel = np.ones((size, size), np.float32) / (size ** 2)
# print(kernel)
#
# dst = cv2.filter2D(image, -1, kernel)
# plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
# plt.show()


# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('test.jpg')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()
#
# dst = cv2.blur(image, (4, 4))
# plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
# plt.show()


import cv2
import matplotlib.pyplot as plt

image = cv2.imread('../rsc/test.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# kernel_size: 홀수
dst = cv2.GaussianBlur(image, (7, 7), 0)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()