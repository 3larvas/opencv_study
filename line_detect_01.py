import cv2
import numpy as np

def roi(equ_frame, vertices):
    mask = np.zeros_like(equ_frame)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(equ_frame, mask)
    return masked

video = cv2.VideoCapture("rsc/mission_left_right.mp4")
# video = cv2.VideoCapture("rsc/mission_bus_lane.mp4")
# video = cv2.VideoCapture("rsc/mission_full.mp4")

while True:
    ret, orig_frame = video.read()
    frame = cv2.GaussianBlur(orig_frame,(5, 5), 0)
    frame = cv2.pyrDown(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # white line detect
    low_w = np.array([0, 0, 130])
    up_w = np.array([255, 80, 255])
    w_mask = cv2.inRange(hsv, low_w, up_w)
    w_edges = cv2.Canny(w_mask, 100, 200)

    # yellow line detect
    low_y = np.array([5, 80, 100])
    up_y = np.array([40, 150, 130])
    y_mask = cv2.inRange(hsv, low_y, up_y)
    y_edges = cv2.Canny(y_mask, 100, 200)

    # blue line detect
    low_b = np.array([120, 130, 70])
    up_b = np.array([170, 170, 120])
    b_mask = cv2.inRange(hsv, low_b, up_b)
    b_edges = cv2.Canny(b_mask, 100, 200)

    # set ROI
    height, width = frame.shape[:2]
    vertices1 = np.array([[(0 , height*0.8), (width * 0.3 , height * 0.6), (width * 0.5, height * 0.6), (width * 0.3, height)]], dtype=np.int32)
    vertices2 = np.array([[(width * 0.7, height), (width * 0.5, height * 0.6), (width * 0.7, height * 0.6), (width, height*0.8)]],dtype=np.int32)
    w_roi_frame1 = roi(w_edges, [vertices1])
    w_roi_frame2 = roi(w_edges, [vertices2])
    y_roi_frame1 = roi(y_edges, [vertices1])
    y_roi_frame2 = roi(y_edges, [vertices2])
    b_roi_frame1 = roi(b_edges, [vertices1])
    b_roi_frame2 = roi(b_edges, [vertices2])
    w_roi_frame = cv2.add(w_roi_frame1, w_roi_frame2)
    w_roi_frame = np.uint8(w_roi_frame)
    y_roi_frame = cv2.add(y_roi_frame1, y_roi_frame2)
    y_roi_frame = np.uint8(y_roi_frame)
    b_roi_frame = cv2.add(b_roi_frame1, b_roi_frame2)
    b_roi_frame = np.uint8(b_roi_frame)

    # draw white lines
    w_lines = cv2.HoughLinesP(w_roi_frame, 1, np.pi/180, 50, maxLineGap=50)
    if w_lines is not None:
        for line in w_lines:
            x1, y1, x2, y2 = line[0]
            if(abs(y1-y2) > height * 0.1  and abs(x1-x2)<width*0.5) :
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)
            elif(abs(y1-y2) < height * 0.025  and abs(x1-x2)>width*0.25) :
                cv2.line(frame, (x1, y1), (x2, y2), (25, 25, 255), 5)

    # draw yellow lines
    y_lines = cv2.HoughLinesP(y_roi_frame, 1, np.pi / 180, 50, maxLineGap=50)
    if y_lines is not None:
        for line in y_lines:
            x1, y1, x2, y2 = line[0]
            if (abs(y1 - y2) > height * 0.1 and abs(x1 - x2) < width * 0.4):
                cv2.line(frame, (x1, y1), (x2, y2), (51, 204, 255), 5)

    # draw blue lines
    b_lines = cv2.HoughLinesP(b_roi_frame, 1, np.pi / 180, 50, maxLineGap=50)
    if b_lines is not None:
        for line in b_lines:
            x1, y1, x2, y2 = line[0]
            if (abs(y1 - y2) > height * 0.1 and abs(x1 - x2) < width * 0.5):
                cv2.line(frame, (x1, y1), (x2, y2), (255, 24, 25), 5)

    #show img
    cv2.imshow("lineframe", frame)
    cv2.imshow("w_roi_frame", w_roi_frame)
    cv2.imshow("y_roi_frame", y_roi_frame)
    cv2.imshow("b_roi_frame", b_roi_frame)
    roi_frame = cv2.add(w_roi_frame, y_roi_frame)
    roi_frame = cv2.add(roi_frame, b_roi_frame)
    cv2.imshow("roi_frame", roi_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()