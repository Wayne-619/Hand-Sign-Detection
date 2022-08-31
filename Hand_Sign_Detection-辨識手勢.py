# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 19:44:43 2022

@author: Wayne
"""

import cv2 
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("C:/Users/Wayne/Desktop/AI_Models/手勢/model/keras_model.h5",
                        "C:/Users/Wayne/Desktop/AI_Models/手勢/model/labels.txt") 

cropset = 20
imgSize = 300

folder = "C:/Users/Wayne/Desktop/RocK"
counter = 0

labels = ["Good","Rock","Fuck","Peace"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']     
        
        imgCrop = img[y-cropset:y+ h+cropset, x-cropset:x+ w+cropset]
        imgDraw = np.ones((imgSize,imgSize,3),np.uint8)*255
        
        imgCropShape = imgCrop.shape
        
        aspectratio = h/w
        
        if aspectratio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop,( wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil(( imgSize - wCal)/2)
            imgDraw[ : , wGap: wCal + wGap] = imgResize
            
            prediction,index = classifier.getPrediction(imgDraw)  
            print(prediction,index)
            
        
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop,(imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil(( imgSize - hCal)/2)
            imgDraw[hGap: hCal + hGap, :] = imgResize
            prediction,index = classifier.getPrediction(imgDraw) 
            
        cv2.putText(imgOutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageDraw", imgDraw)
        
    cv2.imshow("Image",imgOutput)
    key = cv2.waitKey(1)
    
    
    # if key == ord("s"):
    #     counter = counter + 1
    #     cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgDraw)
    #     print(counter)
    
    
    
    
    
    
    