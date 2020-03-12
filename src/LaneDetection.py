import numpy as np
import cv2
# from CurvedLaneDetection import utlis
import utlis
####################################################

cameraFeed= False
# videoPath = 'project_video.mp4'
videoPath = '../rsc/mission_left_right.mp4'
cameraNo= 1
frameWidth= 640
frameHeight = 480

# def roi(equ_frame, vertices):
#     mask = np.zeros_like(equ_frame)
#     cv2.fillPoly(mask, vertices, 255)
#     masked = cv2.bitwise_and(equ_frame, mask)
#     return masked

if cameraFeed:intialTracbarVals = [24,55,12,100] #  #wT,hT,wB,hB
else:intialTracbarVals = [20,60,1,80]   #wT,hT,wB,hB

####################################################


if cameraFeed:
    cap = cv2.VideoCapture(cameraNo)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
else:
    cap = cv2.VideoCapture(videoPath)
count=0
noOfArrayValues =10
global arrayCurve, arrayCounter
arrayCounter=0
arrayCurve = np.zeros([noOfArrayValues])
myVals=[]
utlis.initializeTrackbars(intialTracbarVals)


while True:

    success, img = cap.read()
    #img = cv2.imread('test3.jpg')
    if cameraFeed== False:img = cv2.resize(img, (frameWidth, frameHeight), None)
    imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()

    imgUndis = utlis.undistort(img)
    imgThres,imgCanny,imgColor = utlis.thresholding(imgUndis)
    src = utlis.valTrackbars()

    # height, width = imgThres.shape[:2]
    # vertices1 = np.array(
    #     [[(0, height * 0.8), (width * 0.3, height * 0.6), (width * 0.5, height * 0.6), (width * 0.3, height)]],
    #     dtype=np.int32)
    # vertices2 = np.array(
    #     [[(width * 0.7, height), (width * 0.5, height * 0.6), (width * 0.7, height * 0.6), (width, height * 0.8)]],
    #     dtype=np.int32)
    # w_roi_frame1 = roi(imgThres, [vertices1])
    # w_roi_frame2 = roi(imgThres, [vertices2])
    # w_roi_frame = cv2.add(w_roi_frame1, w_roi_frame2)
    # w_roi_frame = np.uint8(w_roi_frame)


    imgWarp = utlis.perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
    imgWarpPoints = utlis.drawPoints(imgWarpPoints, src)
    imgSliding, curves, lanes, ploty = utlis.sliding_window(imgWarp, draw_windows=True)

    try:
        curverad =utlis.get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        imgFinal = utlis.draw_lanes(img, curves[0], curves[1],frameWidth,frameHeight,src=src)

        # ## Average
        currentCurve = lane_curve // 50
        if  int(np.sum(arrayCurve)) == 0:averageCurve = currentCurve
        else:
            averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        if abs(averageCurve-currentCurve) >200: arrayCurve[arrayCounter] = averageCurve
        else :arrayCurve[arrayCounter] = currentCurve
        arrayCounter +=1
        if arrayCounter >=noOfArrayValues : arrayCounter=0
        cv2.putText(imgFinal, str(int(averageCurve)), (frameWidth//2-70, 70), cv2.FONT_HERSHEY_DUPLEX, 1.75, (0, 0, 255), 2, cv2.LINE_AA)

    except:
        lane_curve=00
        pass

    imgFinal= utlis.drawLines(imgFinal,lane_curve)


    imgThres = cv2.cvtColor(imgThres,cv2.COLOR_GRAY2BGR)
    imgBlank = np.zeros_like(img)
    imgStacked = utlis.stackImages(0.7, ([img,imgUndis,imgWarpPoints],
                                         [imgColor, imgCanny, imgThres],
                                         [imgWarp,imgSliding,imgFinal]
                                         ))

    cv2.imshow("PipeLine",imgStacked)
    cv2.imshow("Result", imgFinal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
