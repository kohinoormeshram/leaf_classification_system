#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 22:22:15 2018

@author: koh_i_noor
"""

import cv2
import numpy as np
import pickle as pk
import pandas as pd
import math as m

name = 's2015_1.jpg'
#reading the image
img1=cv2.imread(name)
        
        #RESIZING image to size (512,512) pi
img= cv2.resize(img1,(512,512))
        
        #GRAYSCALING
imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        #THRESHOLDING
ret,imgthresh=cv2.threshold(imggray,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        #OPENING Operation
kernel = np.ones((8,8),np.uint8)
opening = cv2.morphologyEx(imgthresh, cv2.MORPH_OPEN, kernel)
        
        #INVERSE THRESHOLDING
ret2,threshInv = cv2.threshold(opening,0,255,cv2.THRESH_BINARY_INV)
        
        #EROSION, to remove some of the pre-processing noise
opening2 = cv2.erode(threshInv,kernel,iterations=1)
        
        #NORMALIZE image to Dtype=8U
image = cv2.normalize(src=opening2, dst=None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
        
        #Finding CONTOURS
ret,contours,hierarchy = cv2.findContours(image, 1, 2)
        
        #Finding AREA using Moments(using Contours)
area=cv2.moments(contours[0])['m00']
        
        #Finding PERIMETER
perimeter=cv2.arcLength(contours[0],True)
        
        #Estimating CONVEX HULL
hull = cv2.convexHull(contours[0])
        
        #Finding Hull_Area and Hull_Perimeter
hull_area = cv2.contourArea(hull)
hull_peri=cv2.arcLength(hull,True)
        
        #Making a bounding rectangle using contours and estimating width and height of the box
x,y,w,h = cv2.boundingRect(contours[0])
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)


hist1 = cv2.calcHist([img],[0],None,[256],[0,255])
hist2 = cv2.calcHist([img],[1],None,[256],[0,255])
hist3 = cv2.calcHist([img],[2],None,[256],[0,255])

img = cv2.GaussianBlur(img1,(5,5),0)
imgres = cv2.resize(img,(512,512))
gray_img = cv2.cvtColor(imgres,cv2.COLOR_BGR2GRAY)


sobelx = cv2.Sobel(imgres,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(imgres,cv2.CV_64F,0,1,ksize=5)

mag, angle = cv2.cartToPolar(sobelx,sobely,angleInDegrees=True)

hog = []

for pop in range(0,37):
    hog.append(0)
        
hog = np.array(hog)
    
for k in range(0,64):
    for l in range(0,64):
        x = angle[k][l][0]
        p = m.floor(x/10) * 10
        if p!=36:
            q = p+10
        else:
            q=0
        
        floor_a = x-p
        floor_b = 10-floor_a
        
        hog[p//10] += floor_b * mag[k][l][0]
        hog[q//10] += floor_a * mag[k][l][0]

feat = []
feat.extend(hist1)
feat.extend(hist2)
feat.extend(hist3)
feat.append((float(w)/h))
feat.append((float(area)/(w*h)))
feat.append(float(perimeter)/area)
feat.append(float(hull_peri)/perimeter)
feat.append(float(area)/hull_area)
feat.extend(hog)
#print(w,h,area,perimeter,hull_peri,hull_area)


species = pd.read_csv('species.csv',index_col=0)
species.columns=['name']
l = species['name']

randFor = pk.load(open('Forest_Regression_Classifier.sav', 'rb'))
prediction = randFor.predict([feat])
print(l[prediction[0]])