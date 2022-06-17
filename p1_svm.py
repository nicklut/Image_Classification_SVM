"""
Author:   Nicholas Lutrzykowski
Course:   CSCI 6270
Homework:  4
Problem: 1
File:    p1_svm.py

Purpose: A script that takes descriptors and applies svm

Questions: 
	- What does the confusion matrix mean? It is giving me confusion... 
	- Is the offset bias the same thing as the svc.intercept_ attribute? 
	- How exactly does the validation work? 
		Train over and over using the same data but with different training param? 
		Is the only parameter that changes c? 

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import os
import math
import random
import sys
import copy
import glob

def read_descriptors(in_dir):
	names = ["test.npy", "train.npy", "valid.npy"]

	in_dir = os.path.join(os.getcwd(), in_dir)

	if not os.path.exists(in_dir):
		print("Histogram Shape incorrect!")
		sys.exit()

	descriptors = []
	np_load_old = np.load
	np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

	for name in names:
		with open(os.path.join(in_dir, name), 'rb') as f:
			a = np.load(f)
			descriptors.append(a)

	np.load = np_load_old

	return descriptors[0], descriptors[1], descriptors[2]


def run_svm(train, train_x, train_y, c, prev_success, prev_c, prev_model):
	d_res = [] 
	weights = [] 
	offsets = [] 
	models = [] 

	for i in range(1, 6):
		pos = np.where(train[:, 0] == i)
		y = np.full(pos[0].shape, 1)

		neg = np.where(train[:,0] != i)
		y = np.concatenate((y, np.full(neg[0].shape, -1)))

		#data = np.concatenate((train[pos,1], train[neg,1]))
		data = np.concatenate((train[pos, 1][0], train[neg, 1][0]))
		
		temp = []
		for d in data: 
			temp.append(d)
		data = np.array(temp) 

		
		#X, y = make_classification(n_features = 4, random_state = 0)
		#LSVCClf = LinearSVC(dual = False, random_state = 0, penalty = 'l1',tol = 1e-5)
		#LSVCClf.fit(data, y)
		scaler = StandardScaler()
		scaler.fit(data)
		data = scaler.transform(data)
		model = LinearSVC(C=c)
		model.fit(data, y)

		w = model.coef_
		b = model.intercept_

		dots = [] 
		# Is there a way to do this calculation with numpy??
		
		w_temp = np.reshape(w.T, data[0].shape)
		for x in train_x: 
			dots.append(np.dot(w_temp, x))
		
		dots = np.array(dots)
		dots += b
		d = (1/np.linalg.norm(train_x, axis=1))
		d *= dots

		d_res.append(d)

		weights.append(w)
		offsets.append(b)
		models.append(model)
		

		

	d_res = np.array(d_res)
	results = np.argmax(d_res, axis=0)+1

	

	results -= 1
	train_y -= 1 
	confusion = confusion_matrix(train_y, results)
	confusion = confusion[:5, -5:]
	success = confusion[0,0]+confusion[1,1]+confusion[2,2]+confusion[3,3]+confusion[4,4]
	success_rate = success/np.sum(confusion)
	print("Success Rate: {:.3f}".format(success_rate))
	print("Confusion Matrix for C =", c)
	print(confusion)
	print()

	if success_rate > prev_success:
		return c, success_rate, models
	else:
		return prev_c, prev_success, prev_model

def convert_data(train):
	train_y = [] 
	train_x = [] 

	for i in range(len(train)):
		train_y.append(train[i][0])
		train_x.append(train[i][1])
	train_y = np.array(train_y)
	train_x = np.array(train_x)

	return train_x, train_y

if __name__ == "__main__":

	# Command Line Arguments (python hw4_align.py in_dir out_dir)
	if len(sys.argv) != 2: 
		print("Enter in format: <in_dir>")
		sys.exit()

	in_dir = sys.argv[1]
	
	# Read in training, validation, and training data 
	test, train, valid = read_descriptors(in_dir)

	train_x, train_y = convert_data(train) 
	valid_x, valid_y = convert_data(valid)
	test_x, test_y = convert_data(test)
	

	# Train based on the data set
	# There are 5 different classifications, must train for each one 
	print("Training Results:")
	train_param, success_rate, models = run_svm(train, train_x, train_y, 1, 0, 0, [])
	print()
	# Run validation 
	# Consider changing value of C
	c_val = [.001, .01, 0.1, 0.5, 0.9, 5, 10]

	
	for c in c_val:
		train_param, success_rate, models = run_svm(valid, valid_x, valid_y, c, success_rate, train_param, models)

	print("Final Validation Success Rate: {:.3f}".format(success_rate))
	print("Training Parameter:", train_param)
	
	# Now use the finalized models for test data

	data = test[:, 1]

	temp = []
	for d in data: 
		temp.append(d)
	data = np.array(temp) 
	scaler = StandardScaler()
	scaler.fit(data)
	data = scaler.transform(data)
	results = [] 
	for model in models:
		result = model.predict(data)
		results.append(result)

	results = np.array(results)
	results = np.argmax(results, axis=0)
	train_y -= 1 
	confusion = confusion_matrix(test_y, results)
	confusion = confusion[-5:, :5]
	
	success = confusion[0,0]+confusion[1,1]+confusion[2,2]+confusion[3,3]+confusion[4,4]
	success_rate = success/np.sum(confusion)
	print("Test Success Rate: {:.3f}".format(success_rate))
	print("Confusion Matrix for C =", train_param)
	
	print(confusion)


