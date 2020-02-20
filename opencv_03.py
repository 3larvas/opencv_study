# import cv2
# import matplotlib.pyplot as plt
#
# image = cv2.imread('test.jpg')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.show()
#
# expand = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
# plt.imshow(cv2.cvtColor(expand, cv2.COLOR_BGR2RGB))
# plt.show()
#
# shrink = cv2.resize(image, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
# plt.imshow(cv2.cvtColor(shrink, cv2.COLOR_BGR2RGB))
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# image = cv2.imread('test.jpg')
#
# # 행과 열 정보만 저장합니다.
# height, width = image.shape[:2]
#
# M = np.float32([[1, 0, 50], [0, 1, 10]])
# dst = cv2.warpAffine(image, M, (width, height))
#
# plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
# plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('test.jpg')

# 행과 열 정보만 저장합니다.
height, width = image.shape[:2]

M = cv2.getRotationMatrix2D((width / 2, height / 2), 90, 0.5)
dst = cv2.warpAffine(image, M, (width, height))

plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.show()