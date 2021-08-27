import numpy as np
import cv2
from renderer import plot_sdf
from shape import Polygon
import torch
import random
import json
import math

# get sal/sald data preparation from deepsdf samples

TRAIN_DATA_PATH = '../datasets/train/'
VAL_DATA_PATH = '../datasets/val/'
SALD_TRAIN_PATH='../datasets/sald/train/'
SALD_VAL_PATH='../datasets/sald/val/'

shape_name="shape3"
seed_num=1
sample_num=3000

file_path=shape_name+"_"+str(seed_num)+"_"+str(sample_num)

# get training data
trainData=[]
surface_points=[]
all_sample=[]
f = open(f'{TRAIN_DATA_PATH}{file_path}.txt', 'r')
line = f.readline()
grad_data=[]
while line:
    x, y, sdfx, sdfy = map(lambda n: float(n), line.strip('\n').split(' '))
    if sdfy!=-100:
        surface_points.append([x,y])
        grad_data.append([x, y, sdfx, sdfy])
    else:
        all_sample.append([x,y])
    line = f.readline()
f.close()

# compute gt sdf value for sample points
sampleDist=[]
for i in range(len(all_sample)):
    curP = all_sample[i]
    nearDist = 10000000000000
    for j in range(len(surface_points)):
        curSurP = surface_points[j]
        curDist= math.hypot(curP[0]-curSurP[0],curP[1]-curSurP[1])
        if curDist<nearDist:
            nearDist=curDist
    # get gt dist
    sampleDist.append(nearDist)

# write data
for i in range(len(all_sample)):
    trainData.append([all_sample[i][0],all_sample[i][1],sampleDist[i],-100])
for i in range(len(grad_data)):
    trainData.append(grad_data[i])

f = open(f'{SALD_TRAIN_PATH}{file_path}.txt', 'w')
for datum in trainData:
    f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
f.close()



# get testing data
trainData=[]
surface_points=[]
all_sample=[]
grad_data=[]
f = open(f'{VAL_DATA_PATH}{file_path}.txt', 'r')
line = f.readline()

while line:
    x, y, sdfx, sdfy = map(lambda n: float(n), line.strip('\n').split(' '))
    if sdfy!=-100:
        surface_points.append([x,y])
        grad_data.append([x, y, sdfx, sdfy])
    else:
        all_sample.append([x,y])
    line = f.readline()
f.close()

# compute gt sdf value for sample points
sampleDist=[]
for i in range(len(all_sample)):
    curP = all_sample[i]
    nearDist = 10000000000000
    for j in range(len(surface_points)):
        curSurP = surface_points[j]
        curDist= math.hypot(curP[0]-curSurP[0],curP[1]-curSurP[1])
        if curDist<nearDist:
            nearDist=curDist
    # get gt dist
    sampleDist.append(nearDist)

# write back
for i in range(len(all_sample)):
    testData.append([all_sample[i][0],all_sample[i][1],sampleDist[i],-100])
for i in range(len(grad_data)):
    testData.append(grad_data[i])

f = open(f'{SALD_VAL_PATH}{file_path}.txt', 'w')
for datum in testData:
    f.write(f'{datum[0]} {datum[1]} {datum[2]} {datum[3]}\n')
f.close()
