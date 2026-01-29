# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:54:40 2023
@author: yuhsi
"""

import os
import csv
import random
import numpy as np
from pandas import *

# csv header
header = ['VideoName', 'ShotSeq', 'HitFrame', 'Hitter', 'RoundHead', 'Backhand', 'BallHeight',
              'LandingX', 'LandingY', 'HitterLocationX', 'HitterLocationY', 
              'DefenderLocationX', 'DefenderLocationY', 'BallType', 'Winner']



#data = read_csv(r"C:\Users\user\Badminton\src\TrackNetV2_pytorch\event\00005_predict_denoise_event.csv")
#eventList = data.event.tolist()
#frameList = np.nonzero(eventList)[0]
#print(eventList, frameList)
#print(len(eventList))
#print(np.sum(eventList))



# csv data
data = []
eventPath = r"C:\Users\user\Badminton\src\TrackNetV2-pytorch-option\10-10Gray\event"
eventFolder = sorted(os.listdir(eventPath))

for f in eventFolder:
    if not f.lower().endswith(".csv"):
        continue

    ss = 1
    csv_data = read_csv(os.path.join(eventPath, f))
    eventList = csv_data.event.tolist()
    frameList = np.nonzero(eventList)[0]

    vn = f.split('_')[0] + '.mp4'   # (keeps your original naming logic)

    for fr in frameList:
        data.append([vn, ss, int(fr), 'X', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'X'])
        ss += 1




with open('tracknetv2_pytorch_10-10Gray_denoise_eventDetection_X.csv', 'w', encoding='UTF8', newline='') as f:    ################################ 2. save-name ################################
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
