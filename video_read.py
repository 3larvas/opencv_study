import cv2
import numpy as np

cap = cv2.VideoCapture('red_road_04.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))
    if ret:
        # BGR->HSV로 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # red 영역의 from ~ to
        lower_red_1 = np.array([170, 10, 10])
        upper_red_1 = np.array([255, 255, 255])

        lower_red_2 = np.array([0, 10, 10])
        upper_red_2 = np.array([10, 255, 255])

        # 이미지에서 red영역
        mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        result_mask = cv2.add(mask_1, mask_2)

        # bit연산자를 통해서 red영역만 남김.
        res = cv2.bitwise_and(frame, frame, mask=result_mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', result_mask)
        cv2.imshow('res', res)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
