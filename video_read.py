import cv2
import numpy as np
L_LINE_UP_COL = 240
L_LINE_UP_ROW = 255
L_LINE_DN_COL = 160
L_LINE_DN_ROW = 300

R_LINE_UP_COL = 380
R_LINE_UP_ROW = 255
R_LINE_DN_COL = 460
R_LINE_DN_ROW = 300

cap = cv2.VideoCapture('rsc/red_road_01.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 360))
    if ret:
        # 현재 차선 가이드라인(고정)
        frame = cv2.line(frame, (L_LINE_UP_COL, L_LINE_UP_ROW), (L_LINE_DN_COL, L_LINE_DN_ROW), (255, 0, 0), 3)
        frame = cv2.line(frame, (R_LINE_UP_COL, R_LINE_UP_ROW), (R_LINE_DN_COL, R_LINE_DN_ROW), (255, 0, 0), 3)
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
        cnt = 0
        cnt_255 = 0
        for i in range(240, 380):
            for j in range(255, 300):
                cnt += 1
                if(result_mask[j][i]==255): cnt_255 += 1
        msg =  str(round(cnt_255 * 100 / cnt,2)) + "%"
        cv2.putText(result_mask, msg, (30, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)


        # bit연산자를 통해서 red영역만 남김.
        res = cv2.bitwise_and(frame, frame, mask=result_mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', result_mask)
        cv2.imshow('res', res)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
