
import cv2
import matplotlib.pyplot as plt

image_1 = cv2.imread('../rsc/test.jpg')
image_2 = cv2.imread('lena.png')

result = cv2.add(image_1, image_2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()

result = image_1 + image_2
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.show()