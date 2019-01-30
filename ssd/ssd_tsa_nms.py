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
import glob

#global variables
usematplotDisp = True
figureCount = 1
mSaveOutputImage = True
mProb = 0.1
m_nmsThreshold = 0.4

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

#mImageName = './images/example_03.jpg'
mImageName = '/home/avishek/AnacondaProjects/tsa/data1/EDS_165314_500x500.jpg'
#mImageName = './tsa-data/EDS_20171011_165416_.jpg'
mOutputCounter = 0
mModelName = './coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.caffemodel'
mModelWeightsName ='./coco/SSD_300x300/deploy.prototxt'

SelectedOBjNames =['backpack','suitcase','handbag']

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

def forward_pass_nms(image):
    (h, w) = image.shape[:2]
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    # (note: normalization is done via the authors of the MobileNet SSD
    # implementation)
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
    
    #print("no of detections =", detections.shape)
    confidenceList =[]
    classIds =[]
    boxes = []
    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
    	# prediction
        confidence = detections[0, 0, i, 2]
        # extract the index of the class label from the `detections`,
        idx = int(detections[0, 0, i, 1])
        isCurrentObjectInList = mClassesName[idx] in SelectedOBjNames
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if (confidence > mProb) and (isCurrentObjectInList):
            # compute the (x, y)-coordinates of the bounding box for
    		# the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            confidenceList.append(float(confidence))
            classIds.append(idx)
            (startX, startY, endX, endY) = box.astype("int")
            boxes.append([startX, startY, endX -startX, endY -startY])
            #boxes.append(box)
    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidenceList, mProb, m_nmsThreshold)
    for indx in indices:
        indx = indx
        box = boxes[indx]
        confidence = confidenceList[indx]
        class_id = classIds[indx]
    	# display the prediction
        label = "{}: {:.2f}%".format(mClassesName[class_id], confidence * 100)
        print("[INFO] {}".format(label))
        if mClassesName[idx] in SelectedOBjNames:
            startX, startY, endX, endY = box[0],box[1],box[0]+box[2], box[1]+box[3]
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          COLORS[class_id], 10)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[class_id], 5)

#%% ------------------------------------------
#== single file mode example
img = cv2.imread(mImageName)
forward_pass_nms(img)
#show output image
plt.imshow(convertToRGB(img))
plt.show()
#print("maximum confidence is: ", np.max(confidenceList))
if mSaveOutputImage:
    outputImageName = mImageName.split('/')
    outputImageName = outputImageName[-1][:-4]
    outputFileName = './output/'+ outputImageName +'_out.jpg'
    print('[INFO]  Output file is stored at %s' %(outputFileName))
    cv2.imwrite(outputFileName, img)

#%% ========================================
##load multiple images froma  folder and save them to the output directory
#for  file in glob.glob('./tsa-data/*.jpg'):
#    img = cv2.imread(file)
#    forward_pass_nms(img)
#    # show the output image
#    
#    #plt.imshow(convertToRGB(img))
#    #plt.show()
#    if mSaveOutputImage:
#        outputImageName = file.split('/')
#        outputImageName = outputImageName[-1][:-4]
#        outputFileName = './output/'+ outputImageName +'_out.jpg'
#        print('[INFO]  Output file is stored at %s' %(outputFileName))
#        cv2.imwrite(outputFileName, img)

#cv2.waitKey(0)