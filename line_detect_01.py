import cv2
import numpy as np
import math

def roi(equ_frame, vertices):
    mask = np.zeros_like(equ_frame)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(equ_frame, mask)
    return masked

def nothing(x):
    pass

def initializeTrackbars(intialTracbarVals):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 360, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],50, nothing)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], 100, nothing)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2], 50, nothing)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], 100, nothing)

def valTrackbars():
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")

    src = np.float32([(widthTop/100,heightTop/100), (1-(widthTop/100), heightTop/100),
                      (widthBottom/100, heightBottom/100), (1-(widthBottom/100), heightBottom/100)])
    #src = np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])
    return src

def perspective_warp(img,
                     dst_size=(1280, 720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes
    dst = dst * np.float32(dst_size)
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

intialTracbarVals = [25,60,1,70]   #wT,hT,wB,hB
video = cv2.VideoCapture("rsc/mission_left_right.mp4")

initializeTrackbars(intialTracbarVals)

while True:
    perspective_val = valTrackbars()
    ret, orig_frame = video.read()
    frame = cv2.GaussianBlur(orig_frame,(5, 5), 0)
    height, width = frame.shape[:2]

    frame = perspective_warp(frame, dst_size=(width, height), src=perspective_val)
    # line_img_frame = perspective_warp(frame, dst_size=(width, height), src=perspective_val)
    frame = cv2.pyrDown(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # white line detect
    low_w = np.array([0, 0, 130])
    up_w = np.array([255, 50, 255])
    w_mask = cv2.inRange(hsv, low_w, up_w)
    w_edges = cv2.Canny(w_mask, 100, 200)

    # yellow line detect
    low_y = np.array([5, 80, 100])
    up_y = np.array([40, 150, 130])
    y_mask = cv2.inRange(hsv, low_y, up_y)
    y_edges = cv2.Canny(y_mask, 100, 200)

    # blue line detect
    low_b = np.array([120, 70, 70])
    up_b = np.array([170, 170, 120])
    b_mask = cv2.inRange(hsv, low_b, up_b)
    b_edges = cv2.Canny(b_mask, 100, 200)

    # set ROI
    roi_height, roi_width = w_edges.shape[:2]

    # draw white lines
    w_lines = cv2.HoughLinesP(w_edges, 1, np.pi/180, 50, maxLineGap=50)
    sum_angle = 0
    cnt_angle =0
    if w_lines is not None:
        for line in w_lines:
            x1, y1, x2, y2 = line[0]
            if(abs(y1-y2) > roi_height * 0.45  and abs(x1-x2)<roi_width*0.5) :
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 5)
                tmp_angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
                if(90 <= tmp_angle and tmp_angle <= 180) : tmp_angle -= 180
                sum_angle += tmp_angle * (-1)
                cnt_angle += 1
            elif(abs(y1-y2) < roi_height * 0.1  and abs(x1-x2)>roi_width*0.25) :
                cv2.line(frame, (x1, y1), (x2, y2), (25, 25, 255), 5)

    # draw yellow lines
    y_lines = cv2.HoughLinesP(y_edges, 1, np.pi / 180, 50, maxLineGap=50)
    if y_lines is not None:
        for line in y_lines:
            x1, y1, x2, y2 = line[0]
            if (abs(y1 - y2) > roi_height * 0.1 and abs(x1 - x2) < roi_width * 0.4):
                cv2.line(frame, (x1, y1), (x2, y2), (51, 204, 255), 5)
                tmp_angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
                if (90 <= tmp_angle and tmp_angle <= 180): tmp_angle -= 180
                sum_angle += tmp_angle * (-1)
                cnt_angle += 1

    # draw blue lines
    b_lines = cv2.HoughLinesP(b_edges, 1, np.pi / 180, 50, maxLineGap=50)
    if b_lines is not None:
        for line in b_lines:
            x1, y1, x2, y2 = line[0]
            if (abs(y1 - y2) > roi_height * 0.1 and abs(x1 - x2) < roi_width * 0.5):
                cv2.line(frame, (x1, y1), (x2, y2), (255, 24, 25), 5)
                tmp_angle = math.degrees(math.atan2(x2 - x1, y2 - y1))
                if (90 <= tmp_angle and tmp_angle <= 180): tmp_angle -= 180
                sum_angle += tmp_angle * (-1)
                cnt_angle += 1

    # calculate angle
    if(cnt_angle!=0) :
        print(round(sum_angle / cnt_angle, 2))
        msg = str(round(sum_angle / cnt_angle, 2)) + "'"
        cv2.putText(frame, msg , (30, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    #show img
    cv2.imshow("lineframe", frame)
    cv2.imshow("w_roi_frame", w_edges)
    cv2.imshow("y_roi_frame", y_edges)
    cv2.imshow("b_roi_frame", b_edges)

    roi_frame = cv2.add(w_edges, y_edges)
    roi_frame = cv2.add(roi_frame, b_edges)
    cv2.imshow("roi_frame", roi_frame)
    # cv2.imshow("ori_frame", roi_ori_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()