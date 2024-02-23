#!/usr/bin/env python
# coding: utf-8

# In[7]:


pip install cython


# In[8]:


import numpy as np

def RX(image, window_size):
    # Convert the image to float32
    image = np.float32(image)
    
    # Compute the mean and standard deviation of the image
    mean = np.mean(image)
    std = np.std(image)
    
    # Compute the RX anomaly score for each pixel
    anomaly_scores = np.zeros_like(image)
    for i in range(window_size, image.shape[0] - window_size):
        for j in range(window_size, image.shape[1] - window_size):
            window = image[i-window_size:i+window_size+1, j-window_size:j+window_size+1]
            window_mean = np.mean(window)
            window_std = np.std(window)
            anomaly_scores[i, j] = np.sum((window - window_mean)*2) / (window_std*2 * window.size)
    
    return anomaly_scores


# In[9]:


def LRX(image, window_size):
    # Convert the image to float32
    image = np.float32(image)
    
    # Compute the mean and standard deviation of the image
    mean = np.mean(image)
    std = np.std(image)
    
    # Compute the LRX anomaly score for each pixel
    anomaly_scores = np.zeros_like(image)
    for i in range(window_size, image.shape[0] - window_size):
        for j in range(window_size, image.shape[1] - window_size):
            window = image[i-window_size:i+window_size+1, j-window_size:j+window_size+1]
            window_mean = np.mean(window)
            window_std = np.std(window)
            # Compute the LRX anomaly score using the median absolute deviation
            mad = np.median(np.abs(window - window_mean))
            if mad != 0:
                anomaly_scores[i, j] = np.abs(image[i, j] - window_mean) / (1.4826 * mad)
    
    return anomaly_scores


# In[10]:


import cv2

# Load an image
# image = cv2.imread('/content/person_car_2_lwir.png', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('grey_green_netir_3_lwir.png', cv2.IMREAD_GRAYSCALE)
# Compute the RX and LRX anomaly scores
rx_scores = RX(image, 3)
lrx_scores = LRX(image, 3)


# In[11]:


def RX(image, window_size):
    # Convert the image to float32
    image = np.float32(image)
    
    # Compute the mean and standard deviation of the image
    mean = np.mean(image)
    std = np.std(image)
    
    # Compute the RX anomaly score for each pixel
    anomaly_scores = np.zeros_like(image)
    for i in range(window_size, image.shape[0] - window_size):
        for j in range(window_size, image.shape[1] - window_size):
            window = image[i-window_size:i+window_size+1, j-window_size:j+window_size+1]
            window_mean = np.mean(window)
            window_std = np.std(window)
            # Check if the standard deviation is zero
            if window_std == 0:
                anomaly_scores[i, j] = 0
            else:
                anomaly_scores[i, j] = np.sum((window - window_mean)*2) / (window_std*2 * window.size)
    
    return anomaly_scores


# In[12]:


import cv2

import matplotlib.pyplot as plt


# Load an image
# image = cv2.imread('/content/person_car_2_lwir.png', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('grey_green_netir_3_lwir.png', cv2.IMREAD_GRAYSCALE)
plt.imshow(image)

# Compute the RX and LRX anomaly scores
rx_scores = RX(image, 3)
print(rx_scores)
lrx_scores = LRX(image, 3)
print(lrx_scores)


# In[ ]:




