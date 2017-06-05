# -*- coding: utf-8 -*-

import cv2
import imutils

print(cv2.__version__)

cascade_src = 'cars.xml'

#video_src = 'capturacars1.mp4'
# video_src = 'dataset/video2.avi'
video_src = 'http://192.65.213.243/mjpg/video.mjpg?COUNTER#.WS7AtEnQk34.link'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

count = 0
continuar = True
while continuar:
    continuar, img = cap.read()
    # print "ret {0}".format(continuar)
    count = count + 1
    if (count % 1 == 0):
        count = 0
        if (type(img) == type(None)):
            break
    
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(img, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #gray = cv2.GaussianBlur(gray, (21, 21), 0)

        cars = car_cascade.detectMultiScale(gray, 1.2, 1)
        # cars = car_cascade.detectMultiScale(gray, 1.2, 2)
        print "Found {0} faces!".format(len(cars))

        for (x,y,w,h) in cars:
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
            cv2.rectangle(gray,(x,y),(x+w,y+h),(0,0,255),2)      
    
        #cv2.imshow('video', img)
        cv2.imshow('video', gray)
    
        if cv2.waitKey(33) == 27:
            break

cv2.destroyAllWindows()
