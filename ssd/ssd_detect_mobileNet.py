#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 19:00:40 2019

@author: avishek
"""

# USAGE
# python deep_learning_object_detection.py --image images/example_01.jpg \
#	--prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import time

#global variables
usematplotDisp = True
figureCount = 1
mSaveOutputImage = True
mProb = 0.2

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

mImageName = './images/example_04.jpg'
mModelName = './ssd/mobile-net/MobileNetSSD_deploy.caffemodel'
mModelWeightsName ='./ssd/mobile-net/MobileNetSSD_deploy.prototxt.txt'

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(mModelWeightsName, mModelName)

# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
# (note: normalization is done via the authors of the MobileNet SSD
# implementation)
image = cv2.imread(mImageName)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("[INFO] computing object detections...")
net.setInput(blob)
start_time = time.time()
detections = net.forward()
print("[INFO] ForwardNet pass took %0.3f seconds." %(time.time()-start_time))

# loop over the detections
for i in np.arange(0, detections.shape[2]):
	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

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
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
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
if mSaveOutputImage:
    outputImageName = mImageName.split('/')
    outputImageName = outputImageName[-1][:-4]
    outputFileName = './output/'+ outputImageName +'_out.jpg'
    print('[INFO]  Output file is stored at %s' %(outputFileName))
    cv2.imwrite(outputFileName, image)
#cv2.waitKey(0)