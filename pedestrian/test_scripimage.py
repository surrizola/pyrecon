from SimpleCV import *
import sys
import requests
import json
import numpy as np
import argparse
import glob
import cv2
from datetime import datetime


def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

cam = Camera()

image = cam.getImage()
#fileName = "screenshot_" + datetime.now().strftime("%m-%d_%H:%M") + ".png"
fileName = "screenshot_.png"

image.save(fileName)

# crop with full car & no back wall:
#image = image.crop(170, 170, 230, 300)

# used to use 100, 400 (was the worst)
# followed by 50, 200 which was terrible
# used 50, 400 with success
# used 25, 400 with success
# used 300, 400 with success
#image = image.edges(25, 400)
#image2 = image.edges(100,200)
image2 = image.edges(50,100)

output = image.edges(50, 100)
# generate the side by side image.
#result = halfsies(image,output)
# show the results.
#result.show()
# save the output images. 


#output.save('juniperedges.png')

imagex = cv2.imread(fileName)
gray = cv2.cvtColor(imagex, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

wide = cv2.Canny(blurred, 10, 10)
tight = cv2.Canny(blurred, 225, 250)
auto = auto_canny(blurred)

nzCount = cv2.countNonZero(auto);
#print "NON ZERO AUTO "+ str(nzCount)
 

#auto.save('juniperedges-2.png')
#cv2.imshow("Edges", np.hstack([wide, tight, auto]))
cv2.imshow("Edges",  auto)
cv2.waitKey(0)



# make MASK!
#mask = Image(image2.size())
#dl = DrawingLayer(image2.size())
# get rid of bushes
#dl.polygon([(230, 100), (230, 300), (0, 300), (0, 200)], filled=True, color=Color.WHITE)
# get rid of brick wall
#dl.polygon([(0, 0), (50, 0), (15, 300), (0, 300)], filled=True, color=Color.WHITE)
# get rid of back of car
#dl.polygon([(115, 0), (230, 0), (230, 300), (115, 300)], filled=True, color=Color.WHITE)
#mask.addDrawingLayer(dl)
#mask = mask.applyLayers()

#image2 = image2 - mask



#image.show()
#raw_input()

#image_matrix = image2.getNumpy().flatten()

#image_pixel_count = cv2.countNonZero(image_matrix)

#print "Image " + fileName + " has " + str(image_pixel_count) + " pixels"

#image2.save("canny-" + fileName)

#url = "http://pi-parking.herokuapp.com/updates"
#files = {'update[image]': (fileName, open(fileName, 'rb')), 'update[canny_image]': ("canny-" + fileName, open("canny-" + fileName, 'rb'))}


if nzCount > 1500:
  print "TODO LLENO :( "+str(nzCount)
 
else:
  print "NO HAY NADIE !!! :) "+str(nzCount)
 

#r = requests.post(url, data=status, files=files)
#print r.text



