#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image
#image = cv2.imread('Edges.png')
image = cv2.imread('Region_filling.net.png')

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the image
thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]

# Find the contours of the detected objects
contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Fill the contours with the desired color
cv2.drawContours(image, contours, -1, (255,204,54), 20)

# Display the image
# cv2.imshow('Image', image)
#plt.imshow(image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('region_filling.jpg', image)


# In[4]:


# Display the result
cv2.imshow('Result', image)
cv2.waitKey(0)

# Save the result
cv2.imwrite('result.jpg', image)


# In[ ]:




