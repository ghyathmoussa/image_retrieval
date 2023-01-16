import pandas as pd
import cv2 as cv
import os
from methods import *

# read imegas and add the paths to data frame

db_path = './samples/train/'
db_csv = 'train_data.csv'

classes_path = [db_path + folder_name+'/' for folder_name in os.listdir(db_path)] # path of each image
classes = [folder_name for folder_name in os.listdir(db_path)] # classes of images

images = pd.DataFrame()
images['path'] = [cls + y for cls in classes_path for y in os.listdir(cls)]

# convert images to gray
# store  them in data frame
gray_data = []

for path in images['path']:
    img = cv.imread(path)
    gray = convert_to_gray(img)
    gray_data.append(gray)

images.insert(1,'gray_data',gray_data,allow_duplicates=True)

# add to each image it's class
classes_arr = []
for cls in classes:
    for _ in range(len((os.listdir(db_path + cls)))):
        classes_arr.append(cls)

images['classes'] = classes_arr

# convert to LBP 

lbp_data_frame = []
for gray in  images['gray_data']:
    h,w = gray.shape
    print(h,w)
    lbp_arr = create_zero_matrix(h,w)
    for i in range(1,h-1):
        for j in range(1,w-1):
            lbp_arr[i,j] = lbp(gray,i,j)

    lbp_data_frame.append(lbp_arr)

# Calculate Histogram
print('******************** Histogram *******************')
images['lbp_data'] = lbp_data_frame

hist_data = []

for img in images['lbp_data']:
    print(img.shape)
    hist = calc_his(img)

    hist_data.append(hist)

images['hist_data'] = hist_data

print('*********** Writing to json file')

images.to_json('training_v1.json')

new_images_df = pd.DataFrame()

new_images_df['path'] = images['path']

new_images_df['hist_data'] = images['hist_data']

new_images_df['classes'] = images['classes']

new_images_df.to_json('training.json')

print(images.head())

# printing data
