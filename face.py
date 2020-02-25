import cv2
import numpy as np


cap = cv2.VideoCapture("videos/video_3.hevc")


while True:

    ret, frame = cap.read()
    frame = cv2.resize(frame,(500,500))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    faces = face_cascade.detectMultiScale(gray, 1.5, 25)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    cv2.imshow("frame",frame)

    key = cv2.waitKey(1)
    if key==27:
       cv2.destroyAllWindows()
       break


cap.release()
