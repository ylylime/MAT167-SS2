#from https://www.geeksforgeeks.org/image-reconstruction-using-singular-value-decomposition-svd-in-python/
# import module

from tkinter import Image
import requests     #sudo pip3 install requests

import cv2
import numpy as np
import matplotlib.pyplot as plt

# assign and open image
img1 = Image.open("img1.jpeg")
img2 = Image.open("img2.jpeg")
img3 = Image.open("img3.png")

response1 = requests.get(img1, stream=True)
response2 = requests.get(img2, stream=True)
response3 = requests.get(img3, stream=True)


# with open('image.png', 'wb') as f:
# 	f.write(response.content)

# img = cv2.imread('image.png')

# Converting the image into gray scale for faster
# computation.
# gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculating the SVD
u, s, v = np.linalg.svd(img1, full_matrices=False)

# inspect shapes of the matrices
# print(f"u.shape:{u.shape},s.shape:{s.shape},v.shape:{v.shape}")

print(u.shape)