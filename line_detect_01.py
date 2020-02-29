import cv2
import numpy as np

def roi(equ_frame, vertices):
    mask = np.zeros_like(equ_frame)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(equ_frame, mask)
    return masked

video = cv2.VideoCapture("rsc/red_road_03.mp4")
while True:
    ret, orig_frame = video.read()
    frame = cv2.GaussianBlur(orig_frame,(5, 5), 0)
    frame = cv2.pyrDown(frame)  # 라인표시할 프레임
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    low_w = np.array([0, 0, 170])
    up_w = np.array([255, 80, 255])
    yellow_mask = cv2.inRange(hsv, low_w, up_w)
    edges = cv2.Canny(yellow_mask, 100, 200)
    height, width = frame.shape[:2]
    vertices = np.array([[(0, height), (300, height / 2 + 80), (width - 300, height / 2 + 80), (width, height)]], dtype=np.int32)
    roi_edges = roi(edges, vertices)
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if(abs(y1-y2)>50  and abs(x1-x2)<150) :
                cv2.line(frame, (x1, y1), (x2, y2), (51, 204, 255), 5)

    cv2.imshow("lineframe", frame)
    cv2.imshow("roi_frame", roi_edges)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()