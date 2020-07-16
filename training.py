# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:03:03 2020

@author: user
"""
import sys
import numpy as np
import cv2
import os
import mahotas

samples =  np.empty((0, 100))
responses = []
# 指定要查詢的路徑
yourPath = 'samples/'
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
    image = cv2.imread(yourPath+file,0)
    image=cv2.resize(image,(50,50))
#DETECTION THRESHOLD----------------------------------------------------------------------------------------------------
    T= mahotas.thresholding.otsu(image)
    for k in range(1,50,1):
        for z in range(1,50,1):
            color=image[k,z]
            if (color>T):
                image[k,z]=0
            else:
                image[k,z]=255
    thresh2=image.copy()
    keys = [i for i in range(48, 58)]
    roi_small = cv2.resize(thresh2, (10, 10))
    cv2.destroyWindow('norm')
    cv2.imshow('Numero', image)
    fileType=file.split('.',1)[1]
    fileKey=file.split('_',1)[1]
    key = fileKey
    if key == "0":
        cv2.destroyAllWindows()
    elif key in keys:
        sample = roi_small.reshape((1,100))
        samples = np.append(samples,sample,0)
        responses.append(int(chr(key)))
  else:
    print('OH MY GOD !!')
     
responses= np.array(responses,np.float32)
responses = responses.reshape((responses.size,1)) 
print("training complete")
np. savetxt('data/general_samples.data', samples)
np. savetxt('data/general_responses.data', responses)

