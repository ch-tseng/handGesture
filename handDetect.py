#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
from libraryCH.device.lcd import ILI9341

#----- Your configuration ------------------------
displayDevice = 2  # 1--> LCD monitor  2--> ILI9341 TFT

lcd = ILI9341(LCD_size_w=240, LCD_size_h=320, LCD_Rotate=0)

dislpayType = 1  #1--> Contour  2--> Image
markType = 4  #1--> Draw edge  2-->Box selection  3--> Draw & Box  4--> Convex hulls & defects

numInput = raw_input("Please keyin your gesture number (Enter to skip): ")

lcd = ILI9341(LCD_size_w=240, LCD_size_h=320, LCD_Rotate=270)

def wait():
    raw_input('Press Enter')

def createFolder(pathFolder):
    if(not os.path.exists(pathFolder)):
        os.makedirs(pathFolder)

def writeImage(num, img):
    global imgFolder
    imgFile = ("G{}.png".format(num))
    cv2.imwrite(imgFolder + imgFile, img)

imgFolder = ("imgGesture/{}/".format(numInput))
print ("Images will save to: {}".format(imgFolder))
if(not numInput==""):  createFolder(imgFolder)

cap = cv2.VideoCapture(0)
fgbg = cv2.BackgroundSubtractorMOG()

i = 0
while(True):

    ret, frame = cap.read()
    #th = cv2.GaussianBlur(frame,(5,5),0)
    th = cv2.erode(frame, None, iterations=3)
    th = cv2.dilate(th, None, iterations=3)
    th = fgbg.apply(frame)

    th2 = th.copy()

    contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    if(dislpayType==1):
        empty = np.zeros(th.shape[:2], dtype = "uint8")
        layer = cv2.merge([th, empty, empty])
        markColor = (0,255,0)
    elif(dislpayType==2):
        layer = frame
        markColor = (0,255,0)

    if(len(areas)>0):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
       # print("area={}, w*h={}".format(areas[max_index], w*h))
        if(areas[max_index]>15000):
            if(markType==1 or markType==3):
                cv2.drawContours(layer, cnt, -1, markColor, 2)

            if(markType==2 or markType==3):
                cv2.rectangle(layer,(x,y),(x+w,y+h), markColor,2)

            if(markType==4):
                approx=cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
                hull = cv2.convexHull(approx,returnPoints=True)
                hull2 = cv2.convexHull(approx,returnPoints=False)
                defect = cv2.convexityDefects(approx,hull2)

                #draw the points for the hull 
                if (hull is not None):
                    for i in range ( len ( hull ) ):
                       [x , y]= hull[i][0].flatten()
                       cv2.circle(layer,(int(x),int(y)),2,(0,255,0),-1)
                       cv2.circle(layer,(int(x),int(y)),5,(255,255,0),1)
                       cv2.circle(layer,(int(x),int(y)),8,(255,0,0),1)

                    print ("Convex hull predict: " + str ( len(hull)-2 ))

                #draw the points for the defect
                if (defect is not None):
                    for i in range(defect.shape[0]):
                        s,e,f,d = defect[i,0]
                        start = tuple(approx[s][0])
                        end = tuple(approx[e][0])
                        far = tuple(approx[f][0])
                        cv2.line(layer,start,end,[0,255,0],2)
                        cv2.circle(layer,far,5,[0,0,255],-1)

                    print ("Convex defect predict: " + str ( len(defect)-1 ))

                #mask = np.zeros(layer.shape[:2], dtype ="uint8")
                #cv2.rectangle(mask, (x, y), (h, w), 255, -1)
                #layer = cv2.bitwise_and(layer, layer, mask=mask)

    lcd.displayImg(layer)
    #print(i)

    if(not numInput==""): 
        Cutted = frame[y:y + h, x:x + w]
        layer = layer[y:y + h, x:x + w]
        cv2.imwrite(imgFolder + "color-"+str(i)+".png", Cutted)
        writeImage(i, layer)
    #print("dilated.shape={}".format(dilated.shape))

    if cv2.waitKey(5) == 27 :
        break

    i = i + 1

cap.release()
