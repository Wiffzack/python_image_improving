# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
import math
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from PIL import ImageEnhance

import sys
# from PIL import Image
# import numpy as np
# from matplotlib import pyplot as plt
import math

import binascii
import struct
import scipy
import scipy.misc
import scipy.cluster
import os
import shutil, sys, os, glob, cv2
inFile = sys.argv[1]
outFile = sys.argv[2]
#dst = sys.argv[2]
print (inFile)
 
counter=30
color_saturation=""
 
NUM_CLUSTERS = 5

# stärke die farbe die am wenigsten vorkommt 
# gitb dden richtigen touch
def color_content(image):
	print('reading image')
	#im = Image.open(image)
	im = image.resize((150, 150))      # optional, to reduce time
	ar = np.asarray(im)
	shape = ar.shape
	ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

	print('finding clusters')
	codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
	print('cluster centres:\n', codes)

	vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
	counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

	index_max = scipy.argmax(counts)                    # find most frequent
	peak = codes[index_max]
	colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
	print('most frequent is %s (#%s)' % (peak, colour))

	
	#ImageColor.getcolor("#23a9dd", "RGB")
	h = colour.lstrip('#')
	color_value =  tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
	if color_value[0] < color_value[1] and color_value[0] < color_value[2]:
		print ("rot")
		color_saturation = "red"
		#'red'
	elif color_value[1] < color_value[0] and color_value[1] < color_value[2]:	
		print ("gelb")
		color_saturation = "yellow"
	else:
		sum = color_value[0] + color_value[1] +color_value[2]
		if(sum>680):
			print("white")
			color_saturation = "white"
		else:
			print ("blue")
			color_saturation = "blue"
		
	#check for white	

		
	print('RGB =', color_value)
	return color_saturation


def change_brightness(img, value=30):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	h, s, v = cv2.split(hsv)
	v = cv2.add(v,value)
	v[v > 255] = 255
	v[v < 0] = 0
	final_hsv = cv2.merge((h, s, v))
	img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
	return img

 
def image_entropy(img):
    """calculate the entropy of an image"""
    histogram = img.histogram()
    histogram_length = sum(histogram)
 
    samples_probability = [float(h) / histogram_length for h in histogram]
 
    return -sum([p * math.log(p, 2) for p in samples_probability if p != 0])
 	


def brightness( im_file ):
	from PIL import Image, ImageStat
	im = im_file.convert('L')
	stat = ImageStat.Stat(im)
	return stat.rms[0]


def estimate_noise(img):
	from scipy.signal import convolve2d
	#I = img.convert('L')
	H, W = img.shape
	M = [[1, -2, 1],
	   [-2, 4, -2],
	   [1, -2, 1]]
	sigma = np.sum(np.sum(np.absolute(convolve2d(img, M))))
	sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))
	return sigma
	
for filename in os.listdir(inFile):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".JPG"): 
		os.chdir(inFile)
		filename = os.path.abspath(filename)
		print (filename)
		try:
			img = cv2.imread(filename,1)
			img = cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB)
		except:
			shutil.copy2(filename, inFile + 'temp.jpg')
			img = cv2.imread(myfolder_path + 'temp.jpg')
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			os.remove(myfolder_path + 'temp.jpg')
		real_img = img
		gray_img  = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
		print ("Noise value:")
		noise_lv = estimate_noise(gray_img)
		print(noise_lv)
		if(noise_lv < 8 and noise_lv > 4):
			img = cv2.bilateralFilter(img, 4, 7, 7)
			img = cv2.fastNlMeansDenoisingColored(img, None, 4, 4, 7, 13)
		if(noise_lv < 1 and color_saturation != "white"):
			img = cv2.bilateralFilter(img, 4, 7, 7)
			img = cv2.fastNlMeansDenoisingColored(img, None, 2, 2, 7, 13)	
		alpha = 1.15 # Contrast control (1.0-3.0)
		beta = 0 # Brightness control (0-100)
		
		
		last_entropy = image_entropy(Image.fromarray(img))
		improve_c = 0.05
		while True:
			img = cv2.convertScaleAbs(img, alpha+improve_c, beta=beta)
			new_entropy = image_entropy(Image.fromarray(img))
			if(new_entropy>last_entropy):
				last_entropy = new_entropy
				improve_c = improve_c + 0.05 
			else:
				break
		
		
		img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
		img = change_brightness(img, value=-20)
		img = Image.fromarray(img)
		###################################################
		## now Image
		converter = ImageEnhance.Color(img)
		last_entropy = image_entropy(img)
		improve_c = 0.05
		while True:
			img = converter.enhance(1.1+improve_c)
			new_entropy = image_entropy(img)
			if(new_entropy>last_entropy):
				last_entropy = new_entropy
				improve_c = improve_c + 0.05 
			else:
				break
		
		brightness_value = brightness( img )
		print(brightness_value)
		color_saturation = color_content(img)
		if (brightness> 190 and color_saturation != "white"):
			enhancer = ImageEnhance.Brightness(img)
			new_fac = 1-(190.0/brightness_value)
			img = enhancer.enhance(1-new_fac)	
				
		print (image_entropy(img))
		#color_saturation = color_content(img)
		#layer = Image.new('RGB', img.size, 	color_saturation) # "hue" selection is done by choosing a color...
		#img = Image.blend(img, layer, 0.00005)

		img = np.asarray(img)
		new_name = str(outFile) + str(counter) + ".jpg"
		mpimg.imsave(new_name, img)
		counter=counter+1
    else:
        continue
 
 
