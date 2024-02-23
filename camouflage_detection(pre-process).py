#!/usr/bin/env python
# coding: utf-8

# In[1]:


#NON-LOCAL GLOBAL MEANS:
# importing libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
  
# Reading image from folder where it is stored
#img = cv2.imread('person_car_2_lwir.png')
#img = cv2.imread('hedge_grey_green_netir_0_lwir.png')
#img = cv2.imread('grass_3_lwir.png')
#img = cv2.imread('person_0_lwir.png')
img = cv2.imread('grey_green_netir_3_lwir.png')

  
# denoising of image saving it into dst image
dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
  
# Plotting of source and destination image
plt.subplot(121), plt.imshow(img)
#plt.subplot(122), plt.imshow(dst)
  
plt.show()
# Display the result
# cv2.imshow('dst', img)
# cv2.waitKey(0)

# Save the result
# cv2.imwrite('dst.jpg', img)


# In[2]:


# Display the result
cv2.imshow('Denoise', img)
cv2.waitKey(0)

# Save the result
cv2.imwrite('dst.jpg', img)


# In[3]:


#CANNY EDGE DETECTION 


# In[4]:


import cv2
 
# Read the original image
#img = cv2.imread('person_car_2_lwir.png') 
#img = cv2.imread('hedge_grey_green_netir_0_lwir.png') 
#img = cv2.imread('grass_3_lwir.png')
#img = cv2.imread('person_0_lwir.png')
img = cv2.imread('grey_green_netir_3_lwir.png')

# Display original image
cv2.imshow('dst', img)
cv2.waitKey(0)
 
# Convert to graycsale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Display Sobel Edge Detection Images
cv2.imshow('Sobel X', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobel Y', sobely)
cv2.waitKey(0)
cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
cv2.waitKey(0)
 
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=50) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)
 
cv2.destroyAllWindows()


# In[8]:


#EDGE PRESERVING


# In[5]:


import numpy as np # linear algebra
import cv2
import os 
import matplotlib.pyplot as plt
import random
import math
from PIL import Image


# In[6]:


def conv(img, kernel):
    kH,kW = kernel.shape
    (imH,imW) = img.shape
    new_img = np.zeros(img.shape)
    pad = int((kH-1)/2)
    
    for y in range(imH-kH):
        for x in range(imW-kW):
            window = img[y:y+kH,x:x+kW]
            new_img[y+pad,x+pad] = (kernel * window).sum()
    
    return new_img


# In[7]:


def get_image():
    #DATADIR = "person_car_2_lwir.png"
    #DATADIR = "hedge_grey_green_netir_0_lwir.png"
    #DATADIR = "grass_3_lwir.png"
    #DATADIR = "person_0_lwir.png"
    DATADIR = "grey_green_netir_3_lwir.png"
    
    
    selected = False
    while(selected == False):
        rand = random.randrange(0,len(os.listdir(DATADIR)))
        img_name = (os.listdir(DATADIR))[rand]
        if img_name.endswith("jpg"):
            img = Image.open(os.path.join(DATADIR, (os.listdir(DATADIR))[rand])).convert('L')
            img_arr = np.asarray(img)
            selected = True

    return img_arr


# In[8]:


def sobel_filter(img):
    sobelx = np.array([[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]])

    sobely = np.array([[-1,-2,-1],
                     [0,0,0],
                     [1,2,1]])
    
    kH,kW = sobelx.shape
    (imH,imW) = img.shape
    new_img = np.zeros(img.shape)
    theta = np.zeros(img.shape)
    hsv_img = np.zeros((img.shape[0],img.shape[1],3))
    nms_img = np.zeros(img.shape)
    pad = int((kH-1)/2)
    
    for y in range(imH-kH):
        for x in range(imW-kW):
            window = img[y:y+kH,x:x+kW]
            new_img[y+pad,x+pad] = (((sobelx * window).sum())**2 + ((sobely * window).sum())**2)**0.5
            if(new_img[y+pad,x+pad] > 0):
                if((sobelx * window).sum() == 0):
                    theta[y+pad,x+pad] = 90
                else:
                    theta[y+pad,x+pad] = (math.degrees(math.atan((sobely * window).sum()/(sobelx * window).sum())))%180
                    
                    
                    
                    
    for i in range(imH-1):
        for j in range(imW-1):
            if(theta[i,j] != 0):
                quad = theta[i,j]//22.5
                neighborA = 255
                neighborB = 255

                if quad in [0,8]:
                    neighborA = new_img[i,j-1] #Left
                    neighborB = new_img[i,j+1] #Right
                elif quad in [1,2]:
                    neighborA = new_img[i-1,j-1] #LowerLeft
                    neighborB = new_img[i+1,j+1] #UpperRight
                elif quad in [5,6]:
                    neighborA = new_img[i-1,j+1] #LowerRight
                    neighborB = new_img[i+1,j-1] #UpperLeft
                elif quad in [3,4]:
                    neighborA = new_img[i-1,j] #Down
                    neighborB = new_img[i+1,j] #Up
                if(max([neighborA,new_img[i,j],neighborB]) == new_img[i,j] and new_img[i,j] >= 255 * 0.4):
                    nms_img[i,j] = new_img[i,j]
                    hsv_img[i,j,:] = [theta[i,j],255,255]
                
             
            
            
    return new_img.astype(np.uint8),hsv_img.astype(np.uint8),nms_img.astype(np.uint8)
            

            


# In[9]:


def get_image():
    img = get_image()
    img = cv2.GaussianBlur(img,(3,3),0)


# In[10]:


def sobel_filter(img):
    
    sobel_img,hsv_img,nms_img = sobel_filter(img)
    plt.imshow(sobel_img,cmap='gray')
    plt.imshow(cv2.cvtColor(hsv_img,cv2.COLOR_HSV2RGB))
    plt.imshow(nms_img,cmap='gray')
    plt.title('Regression plt')
    plt.figure()
    plt.figure(figsize=(6,6))
    plt.title("Original Image (grayscale)")
    plt.title('Regression plt')
    plt.imshow(img,cmap='gray')
    plt.imshow(img_array,cmap='gray')
    plt.imshow(new_array,cmap='gray')
    plt.figure()
    plt.title("After applying Sobel Filter")
    plt.imshow(sobel_img,cmap='gray')
    plt.imshow(img_array,cmap='gray')
    plt.imshow(new_array,cmap='gray')
    plt.figure()
    plt.title("After applying NMS and Double Threshold")
    plt.imshow(nms_img,cmap='gray')
    plt.imshow(img_array,cmap='gray')
    plt.imshow(new_array,cmap='gray')
    plt.figure()
    plt.title("Color representation of gradient direction")
    plt.imshow(cv2.cvtColor(hsv_img,cv2.COLOR_HSV2RGB))


# In[14]:


count = 0
blur = cv2.GaussianBlur(img,(3,3),0.5)
plt.figure(figsize=(10,10))
while(True):
    blur = cv2.GaussianBlur(blur,(3,3),0)
    new_blur = cv2.GaussianBlur(blur,(3,3),0)
    count+=1
    plt.subplot(2,3,count)
    plt.imshow((blur-new_blur),cmap="gray")
    if(count == 6):
        break
    blur = new_blur
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




