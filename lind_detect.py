import cv2
import numpy as np


def roi(equ_frame, vertices):
    # blank mask:
    mask = np.zeros_like(equ_frame)
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    # now only show the area that is the mask
    masked = cv2.bitwise_and(equ_frame, mask)
    return masked


video = cv2.VideoCapture("rsc/red_road_01.mp4")
while True:
    ret, orig_frame = video.read()
    # orig_frame = orig_frame[0:590, 0:1300]
    # 가우시안 피라미드 다운샘플링 사용 가로세로 1/2 씩 줄어든 이미지로 변함
    lineframe = cv2.pyrDown(orig_frame)  # 라인표시할 프레임
    frame = cv2.pyrDown(orig_frame)

    # 노이즈제거를 위해 가우시안블러사용
    gaus_frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # 밝기정보추출(히스토그램 평활화)
    gray_frame = cv2.cvtColor(gaus_frame, cv2.COLOR_BGR2GRAY)
    equ_frame = cv2.equalizeHist(gray_frame)

    # 소벨
    """
    sobelX = np.array([[0, 1, 2],
                            [-1, 0, 1],
                            [-2, -1, 0]])
    gx = cv2.filter2D(equ_frame, cv2.CV_32F, sobelX)
    sobelY = np.array([[-2, -1, 0],
                            [-1, 0, 1],
                            [0, 1, 2]])
    gy = cv2.filter2D(equ_frame, cv2.CV_32F, sobelY)
    mag   = cv2.magnitude(gx, gy)
    """
    # edges_frame = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)

    # 캐니
    edges_frame = cv2.Canny(equ_frame, 100, 200)  # canny를 사용하여 에지검출

    # ROI영역설정
    height, width = frame.shape[:2]  # 이미지 높이, 너비
    vertices = np.array([[(0,height),(300, height/2+80), (width-300, height/2+80), (width,height)]], dtype=np.int32)
    roi_frame = roi(edges_frame, [vertices])

    roi_frame = np.uint8(roi_frame)

    lines = cv2.HoughLinesP(roi_frame, 1, np.pi / 180, 50, maxLineGap=50)

    # 검출된 에지에 확률허프변환을 사용하여 직선을 찾는다.
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            cv2.line(lineframe, (x1, y1), (x2, y2), (51, 104, 255), 2)  # 찾은 직선이 보이게 선을 그린다.
    # frame = roi(frame, [vertices])

    cv2.imshow("lineframe", lineframe)
    cv2.imshow("roi_frame", roi_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
video.release()
cv2.destroyAllWindows()