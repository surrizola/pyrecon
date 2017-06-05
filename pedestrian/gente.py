#pedestrianfrom __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# video_src = '../carTracking/cardet2/video1.avi'
# video_src = 'match5-c0.avi'
# video_src = 'terrace1-c0.avi'
# video_src = 'AVSS_AB_Medium_Divx.avi'
video_src = 'http://webcam.st-malo.com/axis-cgi/mjpg/video.cgi?resolution=640x480'

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cap = cv2.VideoCapture(video_src)

cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")

# cascade = cv2.CascadeClassifier("../carTracking/cars.xml")

print "start ..."
# loop over the image paths
# for imagePath in paths.list_images(args["ima==es])

def draw_detections(img, rects, thickness=1):
	for (xA, yA, xB, yB) in rects:		
		cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# for x, y, w, h in rects:
		# the HOG detector returns slightly lager rectangles than the real objects.
		# so we slightly shrink the rectangles to get a nicer output.
		# pad_w, pad_h = int(0.05*w), int(0.05*h)
		# cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 0, 255), thickness)

def detectPeople(image):
	# for imagePath in paths.list_images("images"):
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	# image = cv2.imread(imagePath)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	# orig = image.copy()

  	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.equalizeHist(gray)

	# detect people in the image
	# (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
	(rects, weights) = hog.detectMultiScale(gray, winStride=(8, 8), padding=(32, 32), scale=1.1)
	#(rects, weights) = hog.detectMultiScale(gray, scale=1.2)

	#COnfiguracion origianl para personas
	#rects = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
	
	#rects = cascade.detectMultiScale(gray, 1.15, 1)

	#Para autos la mejor conf
	#rects = cascade.detectMultiScale(gray, 1.02, 1)

    # draw the original bounding boxes
   	# for (x, y, w, h) in rects:
	# 	cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 	print "Detects {0} before. {1} afeter".format(len(rects),len(pick))
	# draw the final bounding boxes
	draw_detections(image,pick)

	# show some information on the number of bounding boxes
	# filename = imagePath[imagePath.rfind("/") + 1:]
	# print("[INFO] {}: {} original boxes, {} after suppression".format( filename, len(rects), len(pick)))
 
 	return image

count = 0
continuar = True
while continuar:
	continuar, image = cap.read()
	if (type(image) == type(None)):
		break
	count = count + 1
	if (count == 30):
		count = 0
		image = detectPeople(image)
		cv2.imshow('video',image)
		if cv2.waitKey(33) == 27:
			break


cv2.destroyAllWindows()
