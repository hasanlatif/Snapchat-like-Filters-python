'''

 This code will work if you have python 3 ,dlib and opencv installed to your system.
 if not just download the setup of python 3 from python.org.
 install it.

 'pip3 install opencv-python' to install opencv
 and use
 pip3 insatall dlib to install dlib
Run this script go the project directory using command prompt and write  ""python alpa_blending_video.py"
'''


import cv2         #importing opencv-python binding
import numpy as np  ##importing numpy
import dlib        ## importing dlib
camera_port = 0    ## camera port at which it is available ,Change to 1 if 0 doesn't work for you.


detector = dlib.get_frontal_face_detector()    ###initializing dlib HOG detector


def dets_to_bb(dets):
    """

    :param dets: dlib rectangle object from HOG detector
    :return: (x,y,w,h)
    """
    x=dets.top()
    y = dets.top()
    w = dets.right() - x
    h = dets.bottom() - y
    return (x,y,w,h)

img2= cv2.imread("emoji.jpg")   ###  Reading images to be placed on video frame

             ### Camera port
cap = cv2.VideoCapture(camera_port)  ## making object for camera
w=cap.get(3)                          ###getting width of frame
h=cap.get(4)                         ##getting height of frame
a="res"+":"+str(w)+'x'+str(h)         ##concatenating as string
#fps = cap.set(cv2.CAP_PROP_FPS,100)

if (cap.get((cv2.CAP_PROP_FPS))==0):    ### if fps is zero return as  N/A
    fps_w ="fps:N/A"
else:
    fps_w=cap.get(cv2.CAP_PROP_FPS)
print("[Info] Intializing camera...")
while True:
    if (cap.isOpened()):
        _,frame = cap.read()      ###Reading frame
        #print("[Info] Frame read")
        cv2.putText(frame,a,(20,20),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))  ## Putting frame's widht and frame height on frame
        cv2.putText(frame, fps_w, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))  ##putting FPS of camera on frame
        dets= detector(frame)               ### detector
        for (i,rect) in enumerate(dets):
            (x,y,w,h ) =dets_to_bb(rect)       ###calling the above defined function.. see above defined function for detail
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

            roi = frame[x:x+w,y:y+h]         ### getting roi for the img2 to place
            rows, cols, channels = roi.shape   ##getting rows and cols of roi to resize the img2
            img2 = cv2.resize(img2,(cols,rows))   ###resizing the img2 according to the size of roi
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)

            img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  ##Converting frame to gray scale.
            ret, mask = cv2.threshold(img2gray, 249,255 , cv2.THRESH_BINARY_INV)    ###thresholding the captured frame. cv2.threshold(gray_scale_frame,lower_range)thresholding,upper_range_thresholding,type_of_thresholding)
            # cv2.imshow("mask",mask)
            mask_inv = cv2.bitwise_not(mask)  ## making inverse of above thresholded image
            img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)  ##bitwise and  between roi and mask_inv
            img2_fg = cv2.bitwise_and(img2,img2,mask=mask)   ##bitwise and between img2(image to be placed on frame) and mask
            dst = cv2.add(img1_bg,img2_fg)     ## alpha blending of image
            #cv2.imshow("dst",dst)
            frame[x:x+w,y:y+h]=dst ###putting  roi to frame


        cv2.imshow("win",frame)  ##showing frame
        if cv2.waitKey(1) & 0xFF == ord('q'):  ##if q is pressed. break the while loop .
            break
    else:
        print("[Info] Unable to open port! change camera_port to 1")
cap.release()  ### Release the cap object
cv2.destroyAllWindows()  ## Destroy all window
