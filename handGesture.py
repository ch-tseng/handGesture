#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from libraryCH.device.lcd import ILI9341
lcd = ILI9341(LCD_size_w=240, LCD_size_h=320, LCD_Rotate=270)

videoDisplay = 2   #1 -> image, 2 -> bw


cap = cv2.VideoCapture(0)
fgbg = cv2.BackgroundSubtractorMOG()

while(True):
    
    ret, frame = cap.read()

    #th = cv2.erode(frame, None, iterations=3)
    #th = cv2.dilate(th, None, iterations=3)
    th = fgbg.apply(frame)

    if(videoDisplay==1):
        layer = frame.copy()

    elif(videoDisplay==2):
        zeros = np.zeros(frame.shape[:2], dtype = 'uint8')
        layer = cv2.merge([zeros, zeros, th])

    th2 = th.copy()
    contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    if(len(areas)>0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        
        approx=cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        hull = cv2.convexHull(approx,returnPoints=True)
        hull2 = cv2.convexHull(approx,returnPoints=False)
        print("hull={}, hull2={}".format(len(hull), len(hull2) ))

        #draw the points for the hull 
        if (hull is not None):
            for i in range ( len ( hull ) ):
               [x , y]= hull[i][0].flatten()
               cv2.circle(layer,(int(x),int(y)),2,(0,255,0),-1)
               cv2.circle(layer,(int(x),int(y)),5,(255,255,0),1)
               cv2.circle(layer,(int(x),int(y)),8,(255,0,0),1)

            #print ("Convex hull predict: " + str ( len(hull)-2 ))

        if(len(hull2) > 3):
            defect = cv2.convexityDefects(approx,hull2)
            #draw the points for the defect
            if (defect is not None):
                for i in range(defect.shape[0]):
                    s,e,f,d = defect[i,0]
                    start = tuple(approx[s][0])
                    end = tuple(approx[e][0])
                    far = tuple(approx[f][0])
                    cv2.line(layer,start,end,[0,255,0],2)
                    cv2.circle(layer,far,5,[0,0,255],-1)

        lcd.displayImg(layer)
