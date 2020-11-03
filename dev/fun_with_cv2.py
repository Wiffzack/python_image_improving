import numpy as np
from matplotlib import pyplot as plt
import sys
import cv2
infile=sys.argv[1]

img = cv2.imread(infile,1)

laplacian = cv2.Laplacian(img,cv2.CV_64F)
s = cv2.convertScaleAbs(laplacian)
#s = cv2.convertScaleAbs(s, 3, beta=0)

scale = 1
delta = 0
ddepth = cv2.CV_64F
src = cv2.GaussianBlur(img, (3, 3), 0)
gray = src
#gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
#abs_grad_x = cv2.convertScaleAbs(grad_x)
#abs_grad_y = cv2.convertScaleAbs(grad_y)
derivation = cv2.add(grad_x,grad_y)
derivation = cv2.convertScaleAbs(derivation)

result1 = cv2.add(s, derivation)
#result1 = cv2.add(result1, s)  

#result1 = cv2.addWeighted(img, 0.9, derivation, 0.1, 0)
#result1 = cv2.addWeighted(result1, 0.8, s, 0.2, 0)

#result1 = cv2.add(img, derivation) 
#result1 = cv2.convertScaleAbs(result1)
cv2.imshow('nier',result1)
cv2.waitKey(0)


sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

#cv2.imshow('nier',laplacian)
#cv2.waitKey(0)

#sub = cv2.subtract(img, s) 
#cv2.imshow('nier',sub)
#cv2.waitKey(0)
#add_img = cv2.add(img,s)
add_img = cv2.addWeighted(img, 0.9, s, 0.1, 0)
#add_img = add_img.clip(min=0)
#cv2.imshow('nier',add_img)
#cv2.waitKey(0)
#cv2.imwrite("img-grad.jpg", image3)
#plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
#plt.title('Laplacian'), plt.xticks([]), plt.yticks([])


#plt.show()
