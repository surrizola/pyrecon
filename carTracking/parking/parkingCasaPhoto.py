# -*- coding: utf-8 -*-

import cv2
import imutils
from imutils import paths
from imutils.object_detection import non_max_suppression

import numpy as np

print(cv2.__version__)

cascade_src = '../cars.xml'
# video_src = '../road.mp4'
# video_src = 'video1.avi'
photo_dir = 'parking_casa'
# video_src = 'dataset/video2.avi'
# video_src = '../AVSS_PV_Medium_Divx.avi'

# cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)


# loop over the image paths
for imagePath in paths.list_images(photo_dir):
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	original = cv2.imread(imagePath)
	img = imutils.resize(original, width=min(800, original.shape[1]))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	cars = car_cascade.detectMultiScale(gray,scaleFactor=1.02, minNeighbors=3)

	# draw the original bounding boxes
	for (x, y, w, h) in cars:
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
		# cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cars])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
		# cv2.rectangle(gray, (xA, yA), (xB, yB), (0, 255, 0), 2)

	# show some information on the number of bounding boxes
	filename = imagePath[imagePath.rfind("/") + 1:]
	print("[INFO] {}: {} original boxes, {} after suppression".format(
		filename, len(rects), len(pick)))

	# show the output images
	cv2.imshow("Before NMS", img)
	# cv2.imshow("After NMS", image)
	cv2.waitKey(0)
