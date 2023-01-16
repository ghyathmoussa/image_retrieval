import cv2 as cv
import numpy as np
import pandas as pd
import os
from methods import *
import json

"""
This project to produce the best 3 image that similar to given image
Created by: Ghyath Moussa
Computer Science Department of Yidldiz Technical University 

"""

"""
####################### Note #########################
# before you run this file go to db.py and run it
# to create training.csv file and get the data from it  

# step 1 prepare images divided to:
#   - calculate the LBP (Local Binary Pattern) of images
#   - normalize data of image
#   - calculate the histogram of image


# If local neighborhood pixel 
# value is greater than or equal
# to center pixel values then 
# set it to 1 else set 0
# calculate the histogram of each test image
# save to json (to save time :) )

"""

training_data = pd.read_json('training.json')

# normalization
normalized_hists = []
for hist in training_data['hist_data']:
    norm = normalize_zero_one(hist)
    normalized_hists.append(norm)

training_data['normalized_hists'] = normalized_hists
# get the test images path
test_path = './samples/test/'

image_paths = [test_path + file_name for file_name in os.listdir(test_path)] # path to each test image

# convert test images to gray
print('Convert test images to gray.............')
gray_data = []

for path in image_paths:
    img = cv.imread(path)
    gray = convert_to_gray(img)
    gray_data.append(gray)

# convert to LBP

print('apply LBP function............')
lbp_data = []
for gray in gray_data:
    h,w = gray.shape
    lbp_arr = np.zeros((h,w),np.uint8)

    for i in range(1,h-1):
        for j in range(1,w-1):
            lbp_arr[i,j] = lbp(gray,i,j)
    
    lbp_data.append(lbp_arr)

# Calculate Histogram
print('Calculate Histogram of test data............')
hist_data = []

for img in lbp_data:
    hist = calc_his(img)
    hist = normalize_zero_one(hist)
    hist_data.append(hist)

hist_data_test = pd.DataFrame()
hist_data_test['hist_data'] = hist_data
hist_data_test.to_json('test.json')

hist_data = pd.read_json('test.json')


# Distance between histograms (Manhattan)

dist_obj = {}

for index,hist_test in hist_data.iterrows():
    print(f'test for image {index+1}')
    dist_obj[index] = []
    for idx,hist_train in training_data.iterrows():
        for i in range(len(hist_test)):
            dist = 0
            for j in range(len(hist_train.normalized_hists)):
                dist+= abs(hist_test.hist_data[i] - hist_train.normalized_hists[j])
            item = {
                'idx':idx,
                'path':hist_train.path,
                'dist':dist
            }
            dist_obj[index].append(item)


dist_json = json.dumps(dist_obj)

with open('test_results.json','w') as file:
    file.write(dist_json)


json_data = pd.read_json('test_results.json')

results = []

for col in json_data.columns:
    json_data.iloc[:,col] = sorted(json_data.iloc[:,col],key=lambda x: x['dist'])
    temp = {
        'test_id':col,
        'path':image_paths[col],
        'matched':[item for item in json_data.iloc[:,col][:3]]
    }
    results.append(temp)


img = cv.imread(results[1]['path'])
ret_1 = cv.imread(results[1]['matched'][0]['path'])
ret_2 = cv.imread(results[1]['matched'][1]['path'])
ret_3 = cv.imread(results[1]['matched'][2]['path'])

cv.imshow('Main',img)
cv.imshow('1',ret_1)
cv.imshow('2',ret_2)
cv.imshow('3',ret_3)

cv.waitKey(0)
cv.destroyAllWindows()