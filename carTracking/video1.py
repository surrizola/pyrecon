###########################################
###     CHILY TRAC
import cv2
import imutils


#src = "http://192.65.213.243/mjpg/video.mjpg"

print "start ..."
cascade_src = 'cars.xml'

car_cascade = cv2.CascadeClassifier(cascade_src)


hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#capture = cv2.VideoCapture(src)
#print "capture .."
##rc,img = capture.read()
#print " tansform ..."
#imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#jpg = Image.fromarray(imgRGB)

#tmpFile = StringIO.StringIO()

#jpg.save(tmpFile,'JPEG')	

#cv2.imshow('video', jpg)
from imutils.object_detection import non_max_suppression




import urllib
import numpy as np


#Detecta personas en la imagen en escala gray y dibuja los cuadrados en el frame
def detecPeople(gray, frame):
    (rects, weights) = hog.detectMultiScale(gray, winStride=(8, 8), padding=(32, 32), scale=1.1 )
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    #print "Detects {0} before. {1} after".format(len(rects),len(pick))
    print "{0} PERSONAS".format(len(pick))
    for (xA, yA, xB, yB) in rects:      
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 0, 255), 2)
    return frame


#Detecta autos en la imagen en escala gray y dibuja los cuadrados en el frame
def detecCars(gray, frame):
    cars = car_cascade.detectMultiScale(gray, 1.3, 1)
    # cars = car_cascade.detectMultiScale(gray, 1.2, 2)
    print "{0} AUTOS".format(len(cars))

    for (x,y,w,h) in cars:
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)      
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)      
    return frame


# Procesa un frame y detecta autos y personas
def processFrame(img):
    print "Procesando un frame"
    frame = imutils.resize(img, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        


    
    gray2 = cv2.equalizeHist(gray)

    frame = detecPeople(gray2, frame)
    frame = detecCars(gray2, frame)


    cv2.imshow('image',frame)
    if cv2.waitKey(1) ==27:
        exit(0)  




# Realiza las detecciones desde un stream de vide (ojo ! no es una imagen estatica, la url debe ser un strem)
def fromLiveFrameVideo(src):
    stream=urllib.urlopen(src)
    bytes=''
    cant = 0
    while True:
        bytes+=stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a!=-1 and b!=-1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            i = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)
            cant = cant + 1
            if cant % 5 == 0:
                processFrame(i)



#lee un una imagen remota en formate stream desd euna url que no es un stream, sirve para las camaras que sacan las imagenes en cgi
def frame_from_static_image_tream(url):
    
    
    stream=urllib.urlopen(url)    
    bytes=''
    finish = False
    image=''
    while not finish:
        bytes+=stream.read(1024)
        a = bytes.find('\xff\xd8')
        b = bytes.find('\xff\xd9')
        if a!=-1 and b!=-1:
            jpg = bytes[a:b+2]
            bytes= bytes[b+2:]
            print "image"
            image = cv2.imdecode(np.fromstring(jpg, dtype=np.uint8),cv2.CV_LOAD_IMAGE_COLOR)                        
            finish = True
    #cv2.imshow('i', image)
    #cv2.waitKey(0)
    return image
    
    


def fromStaticImageStream( url):
    
    while True:
        print "next"
        image = frame_from_static_image_tream(url)
        processFrame(image)
        #cv2.imshow('i', image)
        #cv2.waitKey(0)
        #if cv2.waitKey(1) ==27:
        #    exit(0)  




liveVideoStream = "http://192.65.213.243/mjpg/video.mjpg"
staticImageStream = "http://67.141.206.85/axis-cgi/jpg/image.cgi"

fromLiveFrameVideo(liveVideoStream)
#fromStaticImageStream(staticImageStream)