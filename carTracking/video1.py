
import cv2
import imutils


src = "http://192.65.213.243/mjpg/video.mjpg"
print "start ..."
cascade_src = 'cars.xml'

car_cascade = cv2.CascadeClassifier(cascade_src)

#capture = cv2.VideoCapture(src)
#print "capture .."
##rc,img = capture.read()
#print " tansform ..."
#imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#jpg = Image.fromarray(imgRGB)

#tmpFile = StringIO.StringIO()

#jpg.save(tmpFile,'JPEG')	

#cv2.imshow('video', jpg)




import urllib
import numpy as np
stream=urllib.urlopen(src)

bytes=''
while True:
    bytes+=stream.read(1024)
    a = bytes.find('\xff\xd8')
    b = bytes.find('\xff\xd9')
    if a!=-1 and b!=-1:
        jpg = bytes[a:b+2]
        bytes= bytes[b+2:]
        i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
        frame = imutils.resize(i, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        

        cars = car_cascade.detectMultiScale(gray, 1.2, 1)
        # cars = car_cascade.detectMultiScale(gray, 1.2, 2)
        print "{0} AUTOS".format(len(cars))

        for (x,y,w,h) in cars:
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
            cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)      

        cv2.imshow('i',gray)
        if cv2.waitKey(1) ==27:
            exit(0)  