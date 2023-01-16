import numpy as np
def convert_to_gray(img):
    r,g,b = img[:,:,0],img[:,:,1],img[:,:,2]
    gray = np.round(0.2989 * r + 0.5870 * g + 0.1140 * b)
    return gray

def normalize_zero_one(arr):
    norm_arr = []
    for i in arr:
        temp = (i-min(arr)) / (max(arr)-min(arr))
        norm_arr.append(temp)
    
    return norm_arr

def get_pixel(img,center,x,y):
    new_value = 0
    if img[x,y] >= center:
        new_value = 1
    else:
        new_value = 0
    return new_value


def lbp(img,x,y):
    center = img[x][y]
    
    val_arr = []

    # neighborhoods
    
    val_arr.append(get_pixel(img, center, x-1, y-1)) # top_left
      
    val_arr.append(get_pixel(img, center, x-1, y)) # top
      
    val_arr.append(get_pixel(img, center, x-1, y + 1)) # top_right
      
    val_arr.append(get_pixel(img, center, x, y + 1)) # right
      
    val_arr.append(get_pixel(img, center, x + 1, y + 1)) # bottom_right
      
    val_arr.append(get_pixel(img, center, x + 1, y)) # bottom
      
    val_arr.append(get_pixel(img, center, x + 1, y-1)) # bottom_left
      
    val_arr.append(get_pixel(img, center, x, y-1)) # left

    power_values = [1,2,4,8,16,32,64,128]

    val = 0

    for i in range(len(val_arr)):
        val += (val_arr[i] * power_values[i])
    
    return val


def calc_his(img):
    hist = create_zero_arr(256)
    for i in range(len(img)):
        for j in range(len(img[i])):
            hist[img[i][j]]+=1
    
    return hist


def create_zero_arr(n):
    zeros = [0 for _ in range(n)]
    
    return zeros

def create_zero_matrix(height,width):
    zeros = [[0 for _ in range(width)] for _ in range(height)]
    return zeros