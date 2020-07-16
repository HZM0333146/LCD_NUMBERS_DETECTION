# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 18:25:11 2020

@author: J
"""
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

def getSample(image_path_name):
    image = cv2.imread(image_path_name)
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    items = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = items[0] if len(items) == 2 else items[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            displayCnt = approx
            break
    warped = four_point_transform(gray, displayCnt.reshape(4, 2))
    output = four_point_transform(image, displayCnt.reshape(4, 2))
    thresh1 = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh2 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    src = thresh2
    binary=cv2.Canny(src, 80, 80 * 2)
    k = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary,cv2.MORPH_DILATE,k)
    items = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contr = items[0] if len(items) == 2 else items[1]
    contr = sorted(contr, key=cv2.contourArea, reverse=True)
    digitCnts = []
    for c in contr:
        (x, y, w, h) = cv2.boundingRect(c)
        #print("i:",i,"x:",x,"y:", y,"w:", w,"h:",h,"W*h=",w*h)
        if w*h>1000:
            digitCnts.append(c)
    
    if len(digitCnts)>0 :
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
        x1=0
        y1=0
        w1=0
        h1=0
        for i in range(len(digitCnts)):
            x, y, w, h = cv2.boundingRect(digitCnts[i]);
            #print("i:",i,"x:",x,"y:", y,"w:", w,"h:",h,"W*h=",w*h)
            w1=w1+w
            if i==0:
                x1=x
                y1=y
                h1=h
        cv2.rectangle(thresh1, (x1,y1), (x1+w1, y1+h1), (0, 0, 0), 2)        
        crop_img = thresh1[y1:y1+h1, x1:x1+w1]
        cv2.imwrite('l2_image/contr_analysis_1.png', crop_img)
#-------------------------------------------------------------
        _img = cv2.resize(crop_img, (50, 50), interpolation=cv2.INTER_CUBIC)
        items = cv2.findContours(_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contr = items[0] if len(items) == 2 else items[1]
        contr = sorted(contr, key=cv2.contourArea, reverse=True)
        contr = contours.sort_contours(contr, method="left-to-right")[0]
        imgArray=[]
        for i in range(len(contr)):
            (x, y, w, h) = cv2.boundingRect(contr[i])
            print("i:",i,"x:",x,"y:", y,"w:", w,"h:",h,"W*h=",w*h)
            cv2.rectangle(thresh1, (x,y), (x+w, y+h), (0, 0, 0), 2)
            im = _img[y:y+h, x:x+w]
            imgArray.append(im)
        return imgArray
    else:
        return None
    
img=getSample("l1_image/U0333146_20200409024150.JPG")
if img!=None:
    for i in range(len(img)):
      cv2.imwrite('l3_image/test'+str(i)+'.png', img[i])
else:
    print("None")
