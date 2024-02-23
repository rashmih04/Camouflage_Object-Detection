#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
#from sap import *

def AED(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 20, 200)

    return edges

def AED_F(image):
    # Apply AED to get the edges
    edges = AED(image)

    # Apply Hough Transform to extract the lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    # Check if lines are detected
    if lines is not None:
        # Draw the lines on the original image
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image, (x1,y1), (x2,y2), (36,179,83), 1)

    return image

# Load the image
image = cv2.imread('grey_green_netir_3_lwir.png')


# Apply AED
edges = AED(image)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# Apply AED-F
result = AED_F(image)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)

# Save the result
cv2.imwrite('result.jpg', result)


# In[3]:


import cv2
import numpy as np
#from sap import *

def AED(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 150, 200)

    return edges

def AED_F(image):
    # Apply AED to get the edges
    edges = AED(image)

    # Apply Hough Transform to extract the lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 30)

    # Check if lines are detected
    if lines is not None:
        # Draw the lines on the original image
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image, (x1,y1), (x2,y2), (36,179,83), 1)

    return image

# Load the image
#image = cv2.imread('person_car_2_lwir.png')
image = cv2.imread('Canny Edge Detection_person_car.png')

# Apply AED
edges = AED(image)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# Apply AED-F
result = AED_F(image)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)

# Save the result
cv2.imwrite('result.jpg', result)


# In[19]:


import cv2
import numpy as np
#from sap import *

def AED(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    return edges

def AED_F(image):
    # Apply AED to get the edges
    edges = AED(image)

    # Apply Hough Transform to extract the lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    # Check if lines are detected
    if lines is not None:
        # Draw the lines on the original image
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image, (x1,y1), (x2,y2), (36,179,83), 1)

    return image

# Load the image
image = cv2.imread('grass_3_lwir.png')
#image = cv2.imread('grass_1_lwir.png')



# Apply AED
edges = AED(image)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# Apply AED-F
result = AED_F(image)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)

# Save the result
cv2.imwrite('result.jpg', result)


# In[9]:


import cv2
import numpy as np
#from sap import *

def AED(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 30, 200)

    return edges

def AED_F(image):
    # Apply AED to get the edges
    edges = AED(image)

    # Apply Hough Transform to extract the lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)

    # Check if lines are detected
    if lines is not None:
        # Draw the lines on the original image
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image, (x1,y1), (x2,y2), (36,179,83), 1)

    return image

# Load the image
image = cv2.imread('hedge_grey_green_netir_0_lwir.png')




# Apply AED
edges = AED(image)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# Apply AED-F
result = AED_F(image)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)

# Save the result
cv2.imwrite('result.jpg', result)


# In[7]:


import cv2
import numpy as np
#from sap import *

def AED(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    return edges

def AED_F(image):
    # Apply AED to get the edges
    edges = AED(image)

    # Apply Hough Transform to extract the lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 10)

    # Check if lines are detected
    if lines is not None:
        # Draw the lines on the original image
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image, (x1,y1), (x2,y2), (36,179,83), 30)

    return image

# Load the image
image = cv2.imread('person_1_lwir.png')




# Apply AED
edges = AED(image)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)

# Apply AED-F
result = AED_F(image)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)

# Save the result
cv2.imwrite('result.jpg', result)


# In[ ]:




