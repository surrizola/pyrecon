# -*- coding: utf-8 -*-

import cv2
import imutils
print(cv2.__version__)

cascade_src = 'cars.xml'
# video_src = '../road.mp4'
# video_src = 'video1.avi'
#video_src = '../parkinglot_1_480p.mp4'
# video_src = 'dataset/video2.avi'
# video_src = '../AVSS_PV_Medium_Divx.avi'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

count = 0
continuar = True
while continuar:
    continuar, img = cap.read()
    # print "ret {0}".format(continuar)
    count = count + 1
    if (count % 5 == 0):
        if (type(img) == type(None)):
            break
    
        img = imutils.resize(img, width=min(400, img.shape[1]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        #cars = car_cascade.detectMultiScale(gray,scaleFactor= 1.18, minNeighbors=1)
        cars = car_cascade.detectMultiScale(gray,scaleFactor= 1.09, minNeighbors=1)
        # cars = car_cascade.detectMultiScale(gray, 1.2, 2)
        print "Found {0} faces!".format(len(cars))

        for (x,y,w,h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
    
        cv2.imshow('video', img)
    
        if cv2.waitKey(33) == 27:
            break

cv2.destroyAllWindows()
