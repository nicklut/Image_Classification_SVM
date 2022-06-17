"""
Author:   Nicholas Lutrzykowski
Course:   CSCI 6270
Homework:  4
Problem: 1
File:    p1_descriptors.py

Purpose: A script that find the descriptors of given images 
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import os
import math
import random
import sys
import copy
import glob

def read_images(in_dir):
	#new_dir = os.path.join(os.getcwd(), "hw4_data")
	old_dir = os.getcwd()
	new_dir = os.path.join(old_dir, in_dir)
	image_list = [] 
	#gray_list = [] 
	image_names = []

	temp = glob.glob(new_dir + '/*.JPEG')
	for filename in temp: #assuming gif
		#gray_list.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(float))
		image_list.append(cv2.imread(filename))
		image_names.append(filename)

	os.chdir(old_dir)

	return image_list, image_names

def get_discriptors(images, t, bw, bh, i):
	descriptors = [] 
	for img in images:
		H = img.shape[0]
		W = img.shape[1]

		deltaw = int(W/(bw+1))
		deltah = int(H/(bh+1))
		#print(img)
		#print(img[H-1, W-1])

		descriptor = [] 
		for w in range(bw):
			x = w*deltaw
			for h in range(bh):  
				y = h*deltah
				block = img[y:y+(2*deltah), x:x+(2*deltaw)]
				pixels = block.reshape((2*deltah)*(2*deltaw), 3)
				hist, _ = np.histogramdd(pixels, (t,t,t))

				if (hist.shape != (t,t,t)):
					print("Histogram Shape incorrect!")
					sys.exit()

				if (np.sum(hist) != (2*deltah)*(2*deltaw)):
					print("ERROR")
					sys.exit()


				descriptor.append(hist)

		descriptor = np.array(descriptor)
		descriptor = descriptor.flatten()

		descriptors.append(np.array([i+1, np.array(descriptor)]))
	
	return descriptors

if __name__ == "__main__":

	# Command Line Arguments (python hw4_align.py in_dir out_dir)
	if len(sys.argv) != 6: 
		print("Enter in format: <in_dir> <out_dir> <t> <bw> <bh>")
		sys.exit()

	in_dir = sys.argv[1]
	out_dir = sys.argv[2]
	t = int(sys.argv[3])
	bw = int(sys.argv[4])
	bh = int(sys.argv[5])

	# save the current directory 
	old_dir = os.getcwd()

	# Idea: preserve folder structre, just replace all images in each 
	#		folder with a single file containing the descriptors. 

	image_type = [os.path.join(in_dir, "test"), os.path.join(in_dir, "train"), os.path.join(in_dir, "valid")]
	image_out_type = [os.path.join(out_dir, "test"), os.path.join(out_dir, "train"), os.path.join(out_dir, "valid")]
	names = ["test.npy", "train.npy", "valid.npy"]
	# Labels: 1: "grass", 2: "ocean", 3: "redcarpet", 4: "road", 5: "wheatfield"
	image_classifications = ["grass", "ocean", "redcarpet", "road", "wheatfield"]

	# Iterate of the test, train, valid folders

	for i in range(len(image_type)):
		output = [] 
		count = 0
		# Iterate over the classifications and extract all images
		for img_folder in image_classifications:
			in_dir = os.path.join(image_type[i], img_folder)
			image_list, image_names = read_images(in_dir)
			
			# Each image is structured as [img_label, descriptor]
			descriptors = get_discriptors(image_list, t, bw, bh, count)

			output += descriptors
			count += 1

		# Structure is an array of all descriptors of images in that type of folder 
		# Each "descriptor" has structure of [label, descriptor]
		output = np.array(output)

		# Output the results to a file 
		out_dir = os.path.join(os.getcwd(), out_dir)
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)

		with open(os.path.join(out_dir, names[i]), 'wb') as f: 
			f.seek(0)
			f.truncate()
			np.save(f, output)









			



			
