# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 00:32:14 2020

@author: user
"""
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import matplotlib.pyplot as plt
from skimage import measure,draw 
import os
def imgThreshold(img):
    rosource,binary=cv2.threshold(img,121,255,cv2.THRESH_BINARY)
    return binary

def verticalCut(img,img_num):
    (x,y)=img.shape
    pointCount=np.zeros(y,dtype=np.float32)#每列黑色的个数
    x_axes=np.arange(0,y)
    tempimg=img.copy()
    for i in range(0,y):
        for j in range(0,x):
            if(tempimg[j,i]==0):
                pointCount[i]=pointCount[i]+1
    start = []
    end = []
    print(pointCount)
    for index in range(1, y-1):
        if ((pointCount[index-1] == 0) & (pointCount[index] != 0)):
            start.append(index)
        elif ((pointCount[index] != 0) & (pointCount[index +1] == 0)):
            end.append(index)
    imgArr=[]
    for idx in range(0,len(start)):
        tempimg=img[ :,start[idx]:end[idx]]
        cv2.imshow(str(img_num)+"_"+str(idx), tempimg)
        cv2.imwrite(img_num+'_'+str(idx)+'.jpg',tempimg)
        imgArr.append(tempimg)
        print(idx)
    return imgArr

def horizontalCut(img):
    (x,y)=img.shape
    pointCount=np.zeros(y,dtype=np.uint8)
    x_axes=np.arange(0,y)
    for i in range(0,x):
        for j in range(0,y):
            if(img[i,j]==0):
                pointCount[i]=pointCount[i]+1
    start=0
    end=0
    print(pointCount)
    for index in range(1,y):
        if((pointCount[index]!=0)&(pointCount[index-1]==0)):
            start.append(index)
        elif((pointCount[index]==0)&(pointCount[index-1]!=0)):
             end.append(index)
    img1=img[start:end,:]
    cv2.imshow(str("m"),img1)
    return img1
 
def matchTemplate(src,matchSrc,label):
    binaryc=imgThreshold(src)
    result=cv2.matchTemplate(binaryc,matchSrc,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    tw,th=matchSrc.shape[:2]
    tl=(max_loc[0]+th+2,max_loc[1]+tw+2)
    cv2.rectangle(src,max_loc,tl,[0,0,0])
    cv2.putText(src,label,max_loc,fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=0.6,color=(240,230,0))
    cv2.imshow('001',src)
    
def canny_demo(image):
    t = 80
    canny_output = cv2.Canny(image, t, t * 2)
    #cv2.imshow("canny_output", canny_output)
    #cv2.imwrite("canny_output.png", canny_output)
    return canny_output

def getNumberSample(image_file_path_name):
    image = cv2.imread(image_file_path_name)
    image = cv2.resize(image, (500, 500), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200, 255)
    cv2.imwrite('edge.png', edged)
    #(_,cnts,_) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts= imutils.grab_contours(cnts)
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
    cv2.imwrite('warped.png', warped)
    output = four_point_transform(image, displayCnt.reshape(4, 2))
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #cv2.imwrite('thresh1.png', thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    #cv2.imwrite('thresh2.png', thresh)
    """
    _,cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        #print("x:",x,"y:", y,"w:", w,"h:",h,"W*h=",w*h)
        if w*h>=40:
            digitCnts.append(c)   
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    for cnn in digitCnts:
        x, y, w, h = cv2.boundingRect(cnn)
    """
    src = thresh
    binary = canny_demo(src)
    k = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, k)
    #----------------------------------------------------------------------------------
    out, contr, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = []
    for c in contr:
        (x, y, w, h) = cv2.boundingRect(c)
        #print("x:",x,"y:", y,"w:", w,"h:",h,"W*h=",w*h)
        if w*h>=40:
            digitCnts.append(c)
    if len(digitCnts)>0 :
        digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]
    for c in range(len(digitCnts)):
        x, y, w, h = cv2.boundingRect(digitCnts[c]);
        cv2.rectangle(src, (x,y), (x+w, y+h), (0, 0, 0), 2)
        crop_img = src[y:y+h, x:x+w]
        cv2.imwrite('samples/'+""+"_"+str(c)+'.png', crop_img)
    cv2.imwrite("contours_analysis.png", src)
    
#指定要查詢的路徑
yourPath = 'image/'
# 列出指定路徑底下所有檔案(包含資料夾)
allFileList = os.listdir(yourPath)
# 逐一查詢檔案清單
for file in allFileList:
#   使用isdir檢查是否為目錄
#   使用join的方式把路徑與檔案名稱串起來(等同filePath+fileName)
  if os.path.isdir(os.path.join(yourPath,file)):
    print("I'm a directory: " + file)
  elif os.path.isfile(yourPath+file):
    print(yourPath+file)
    getNumberSample(yourPath+file)
  else:
    print('OH MY GOD !!')






