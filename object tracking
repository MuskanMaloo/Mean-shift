import cv2
import numpy as np
cap=cv2.VideoCapture("https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4")
ret,frame=cap.read()
x,y,w,h=300,200,100,50
track_window=(x,y,w,h)
roi=frame[y:y+h,x:x+w]
roi_hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
mask=cv2.inRange(roi_hsv,np.array((0,60,32)),np.array((180,255,255)))
roi_hist=cv2.calcHist([roi_hsv],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,cv2.NORM_MINMAX)
TERM_CRIT=(cv2.TERM_CRITERIA_COUNT,10,1)
while cap.isOpened():
    ret,frame=cap.read()
    if ret==True:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst=cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret,track_window=cv2.meanShift(dst,track_window,TERM_CRIT)
        x,y,w,h=track_window
        final_image=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        cv2.imshow("image",final_image)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
    
