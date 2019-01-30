#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:39:14 2019

@author: avishek
"""

# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import time

#global variables
usematplotDisp = True
figureCount = 1
mSaveOutputImage = False
mProb = 0.4

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def imShow(img, windowName ='newWindow', newWindow= False):
    global figureCount
    if(usematplotDisp):
        if(newWindow):
            figureCount += 1 #increment count
        plt.ion()
        #plt.figure(1)
        if(img.ndim is 3):
            plt.imshow(convertToRGB(img))
        else:
            plt.imshow(img)
        plt.title(windowName)
        plt.show()
    else:
        #here we use opencv
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(windowName, img)


# manually feed the input params and image

mImageName = './images/example_03.jpg'
mImageName = '/home/avishek/AnacondaProjects/tsa/data1/EDS_165314_500x500.jpg'
mImageName = '/home/avishek/AnacondaProjects/tsa/data1/EDS_20171011_165416_.jpg'
mModelName = './coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel'
mModelWeightsName ='./coco/SSD_300x300/deploy.prototxt'

## initialize the list of class labels MobileNet SSD was trained to
## detect, then generate a set of bounding box colors for each class
#CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#	"sofa", "train", "tvmonitor"]
mClassesFile ='./coco/coco.names'
mClassesName =[] 
with open(mClassesFile,'r') as f:
    mClassesName = [ line.strip() for line in f.readlines()]

#account for background for sdd_coco
mClassesName.insert(0, 'background')
print("no of classes =", len(mClassesName))
COLORS = np.random.uniform(0, 255, size=(len(mClassesName), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(mModelWeightsName, mModelName)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread(mImageName)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                             1, (300, 300), (104,117,123), True)
#mean = (104,117,123) from cafee sdd github page 
# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
start_time = time.time()
detections = net.forward('detection_out')
print("[INFO] ForwardNet pass took %0.3f seconds." %(time.time()-start_time))

print("no of detections =", detections.shape)
confidenceList =[]
# loop over the detections
for i in np.arange(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
	# prediction
    confidence = detections[0, 0, i, 2]
    confidenceList.append(confidence)
    # filter out weak detections by ensuring the `confidence` is
    # greater than the minimum confidence
    if confidence > mProb:
        # extract the index of the class label from the `detections`,
		# then compute the (x, y)-coordinates of the bounding box for
		# the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
		# display the prediction
        label = "{}: {:.2f}%".format(mClassesName[idx], confidence * 100)
        print("[INFO] {}".format(label))
        cv2.rectangle(image, (startX, startY), (endX, endY),
			COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

		
# show the output image
#cv2.imshow("Output", image)
plt.imshow(convertToRGB(image))
plt.show()

print("maximum confidence is: ", np.max(confidenceList))
#cv2.waitKey(0)