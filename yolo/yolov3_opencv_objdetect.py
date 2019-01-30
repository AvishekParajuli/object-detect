#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
##============================================
Object detection using yolov3 and opencvdnn
##============================================
@author: avishek
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import sys
import os.path
import time
import argparse

#global variables
usematplotDisp = True
figureCount = 1

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold

#default config and weights
mModelCfg = 'model/yolov3.cfg'
mModelWeightsFile = 'model/yolov3.weights'
#mModelCfg = 'yolov3-tiny.cfg'
#mModelWeightsFile = 'yolov3-tiny.weights'

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

mClassesFile ='./model/coco.names'
mClassesName =[]
with open(mClassesFile,'r') as f:
    mClassesName = [ line.strip() for line in f.readlines()]

RAND_COLOR = np.random.uniform(0, 255, size=(len(mClassesName),3))

mNet = cv2.dnn.readNetFromDarknet(mModelCfg, mModelWeightsFile)
mNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
mNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Get the names of the output layers
def getOutputLayerNames():
    # Get the names of all the layers in the network
    layersNames = mNet.getLayerNames() # this return list
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    #net.getUnconnectedOutLayers() returns an array of index, [[200],[227]]
    return [layersNames[idx[0] - 1] for idx in mNet.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(img, classId, confidence, left, top, right, bottom):
    # Draw a bounding box.
    loc_color = RAND_COLOR[classId]
    cv2.rectangle(img, (left, top), (right, bottom), loc_color, 3)
    # Get the label for the class name and its confidence
    label = '%s: %.2f' % (mClassesName[classId], confidence)
    cv2.putText(img, label, (left-10,top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, loc_color, 2)
    #Display the label at the top of the bounding box
    #labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    #top = max(top, labelSize[1])
    #cv2.rectangle(img, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv2.FILLED)
    #cv2.putText(img, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)




def processImage(img):
    if(img is None):
        print( " Input image could not be read; Exiting from program")
        return
    
    mImgHeight,mImgWidth,_ = img.shape
    mScale = 1/255.0
    blob = cv2.dnn.blobFromImage(img, mScale, (416,416), (0,0,0), True, crop=False)
    mNet.setInput(blob)
    outputList = mNet.forward(getOutputLayerNames())
    classIds = []
    confidences = []
    objConfidenceList =[]
    boxes = []
    #print("length of outputlist", len(outputList))
    for output in outputList:
        #print("length of current output =", len(output))
        for detection in output:
            scores = detection[5:-1]
            classId = np.argmax(scores)
            objConfidence = detection[4]
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * mImgWidth)
                center_y = int(detection[1] * mImgHeight)
                width = int(detection[2] * mImgWidth)
                height = int(detection[3] * mImgHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                objConfidenceList.append(float(objConfidence))
                boxes.append([left, top, width, height])
                
    #print("Before: No of boxes = %i before Non-maxima supression" %len(boxes))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    #print("After: No of boxes = %i After Non-maxima supression" %len(indices))
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        drawPred(img, classIds[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        #print(" Object no: %i,  objconfidence = %f, class confidence= %f " %(i,objConfidenceList[i],confidences[i]))
    # end of for


# end of processimage
def processVideo(videoFileName):
    cap = cv2.VideoCapture(videoFileName)
    if not cap.isOpened():
        print("Video could not be opened")
        return
    print("Succesful in opening video file")
    frameCount = 0
    outputFile = videoFileName[:-4] #strip out .mp4 or .avi
    outputFile = outputFile.split('/') #strip out all the folder separation
    outputFile = outputFile[-1]# only take the videoFileName
    outputFile = './output/' +outputFile +'_yolo_out.avi'
    print('output file is stored at %s' %(outputFile))
    frameList =[]
    processingTimeList =[]
    # initilize video_writer
    vid_writer = cv2.VideoWriter(outputFile, 
                                 cv2.VideoWriter_fourcc('M','J','P','G'),
                                 30, 
                                 (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while True:
        ret, frame = cap.read()
        
        #if end of frame reached prepare to close and write
        if not ret:
            print("End of frame reached")
            #cv2.waitKey(1000)
            cap.release()
            break
        frameCount += 1 #increment count
        start_time = time.time()
        # detect objects and get results on current extracted frame
        processImage(frame)
        processingTimeList.append(time.time()-start_time)
        frameList.append(frame)
        imShow(frame, "video")
        plt.pause(0.001)
    #write the output to the output video
    for frame in frameList:
        vid_writer.write(frame.astype(np.uint8))
    print(" mean = %f, max = %f, min = %f" %(np.average(processingTimeList), np.max(processingTimeList),np.min(processingTimeList)))
    print("exited from whie loop")
    print("total no of frames = ", frameCount)
    vid_writer.release()
    plt.close()

            

    
        
if __name__ == "__main__":
    import sys
    #fib(int(sys.argv[1]))
    if (len(sys.argv)> 1 ):
        parser = argparse.ArgumentParser()
        parser.add_argument('-i',"--image",help ='path to image')
        parser.add_argument('-v', "--video", help = 'path to video')
        #parser.add_argument('-w', "--weights", help = 'path to weights file')
        #parser.add_argument('-c', "--cfg", help = 'path to config file')
        args = parser.parse_args()
        
#        if(args.cfg):
#            if not os.path.exists(args.cfg):
#                print("Input config file ", args.cfg, " doesn't exist")
#                sys.exit(1)
#            mModelCfg = args.cfg
#        if(args.weights):
#            if not os.path.exists(args.weights):
#                print("Input model weights file ", args.weights, " doesn't exist")
#                sys.exit(1)
#            mModelWeightsFile = args.weights
        
        if (args.image):
            if not os.path.exists(args.image):
                print("Input image file ", args.image, " doesn't exist")
                sys.exit(1)
            img = cv2.imread(args.image, cv2.IMREAD_COLOR)
            processImage(img)
            imShow(img)
            time.sleep(5)
            inputFileName = args.image[:-4]#strip out .jpg
            outputFileName = inputFileName.split('/')
            outputFileName = outputFileName[-1]
            outputFileName = './output/'+ outputFileName +'_out.jpg'
            print('output file is stored at %s' %(outputFileName))
            cv2.imwrite(outputFileName, img)
            plt.close()
        elif (args.video):
            if not os.path.exists(args.video):
                print("Input video file ", args.video, " doesn't exist")
                sys.exit(1)
            print("video ")
            processVideo(args.video)
        else:
            print("Invalid inputs; please check the names of images or video file")
    else:
         print("Invalid number of input argumanets; atleas 2 required")
