#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 22:49:44 2018

@author: koh_i_noor
"""

import cv2
import pandas as pd
import os


L=[]
index=[]

src = r"/Users/koh_i_noor/Documents/MiniProject/Folio Leaf Dataset/Folio"
t = '/'
count=0

dirs=os.listdir(src)
dirs.sort()
for i in dirs:
    dirs2=os.listdir(src+t+i)
    dirs2.sort()
    for j in dirs2:
        index.append(count)
        name = src+t+i+t+j
        image = cv2.imread(name)
        im1 = cv2.resize(image,(512,512))
        im1 = cv2.cvtColor(im1,cv2.COLOR_BGR2RGB)

        hist1 = cv2.calcHist([im1],[0],None,[256],[0,255])
        hist2 = cv2.calcHist([im1],[1],None,[256],[0,255])
        hist3 = cv2.calcHist([im1],[2],None,[256],[0,255])

        l=[]
        for x in hist1:
            l.extend(x)
        for x in hist2:
            l.extend(x)
        for x in hist3:
            l.extend(x)
        print(l)
        L.append(l)
    count+=1

df=pd.DataFrame(L,index=index)

column = []
for i in range(0,768):
    column.append('c'+str(i))

df.columns=column
df.to_csv('histo.csv')

df2 = pd.DataFrame(index)
df2.to_csv('index.csv')
