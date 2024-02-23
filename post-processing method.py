#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2

image = plt.imread('grey_green_netir_3_lwir.png')
plt.imshow(image)


# In[2]:


image = cv2.imread('Edges_net.png')
plt.imshow(image)


# In[3]:


import cv2
import numpy as np

# Load the image
image = cv2.imread('Edges_net.png')

# Convert the image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a threshold to the image
thresholded_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)[1]

# Find the contours of the detected objects
contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Fill the contours with the desired color
cv2.drawContours(image, contours, -1, (255,204,51), 20)

# Display the image
cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('net_region_result.jpg', image)


# In[4]:


image = cv2.imread('Region_filling.net.png')
plt.imshow(image)


# In[5]:


import cv2
import numpy as np

# Read the image files.
image1 = cv2.imread("grey_green_netir_3_lwir.png")
# plt.imshow(image1)
image2 = cv2.imread("Edges_net.png")
# plt.imshow(image2)

# Add the images.
added_image = cv2.add(image1, image2)

# Subtract the images.
subtracted_image = cv2.subtract(image2, image1)

# Multiply the images.
multiplied_image = cv2.multiply(image1, image2)

# Divide the images.
divided_image = cv2.divide(image1, image2)

# Save the output images.
# cv2.imwrite("added_image.jpg", added_image)
# cv2.imwrite("subtracted_image.jpg", subtracted_image)
# cv2.imwrite("multiplied_image.jpg", multiplied_image)
# cv2.imwrite("divided_image.jpg", divided_image)


# In[6]:


plt.imshow(image1)


# In[7]:


plt.imshow(image2)


# In[8]:


plt.imshow(added_image)


# In[9]:


plt.imshow(subtracted_image)


# In[10]:


added_image = cv2.subtract( added_image,subtracted_image)
plt.imshow(added_image)


# In[11]:


plt.imshow(multiplied_image)


# In[12]:


plt.imshow(divided_image)


# In[24]:


import numpy as np
import matplotlib.pyplot as plt

# Read the single-band image
image = plt.imread('grey_green_netir_3_lwir.png')

# Define the spectral signature of the target
target_signature = [0.47] # Example target spectral signature

# Set a threshold for target detection
threshold = 0.1  # Example threshold value

# Initialize an empty binary mask for target detection
target_mask = np.zeros_like(image, dtype=bool)

# Iterate over each pixel in the image
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        pixel_value = image[i, j]  # Retrieve the pixel value

        # Calculate similarity with the target signature
        similarity = abs(pixel_value - target_signature[0])

        # Compare similarity with the threshold
        if similarity <= threshold:
            target_mask[i, j] = True  # Mark the pixel as a potential target

# Display the target detection results
plt.figure()
plt.imshow(image, cmap='gray')
plt.imshow(target_mask, alpha=0.4, cmap='Reds')
plt.show()
plt.savefig('target_detection_result.png')


# In[25]:


# Display the result
cv2.imshow('Result', image)
cv2.waitKey(0)
plt.savefig('target_detection_result.png')
# Save the result
cv2.imwrite('result.jpg', image)


# In[ ]:





# In[ ]:





# In[ ]:




