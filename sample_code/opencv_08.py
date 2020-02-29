# 테투리 감지
import cv2
import matplotlib.pyplot as plt



cap = cv2.VideoCapture('../rsc/red_road_03.mp4')

while(cap.isOpened()):
    ret, image = cap.read()
    image = cv2.resize(image, (640, 360))
    if ret:

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 120, 255, 0)

        # plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB))
        # plt.show()

        contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        image = cv2.drawContours(image, contours, -1, (0, 255, 0), 4)

        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        cv2.imshow('frame', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()