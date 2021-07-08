#!/usr/bin/env python
# coding: utf-8

# In[1]:

# In[2]:

import os
import sys
from os import walk, path, listdir
from os.path import isfile, join

import matplotlib.image as img
from PIL import Image, ImageDraw
import math
import numpy as np
import random
import matplotlib.pyplot as plt

#get folder name from cmd
folder_name = sys.argv[1]

#get current directory 
current_dir = os.getcwd()

#find phone directory
find_phone_dir = current_dir + str(folder_name)

#get the file names in the find phone directory
filenames = [f for f in listdir(find_phone_dir) if isfile(join(find_phone_dir, f))]

#get the image names
labels_file = filenames.pop(len(filenames)-1)

#number of samples
num_samples = len(filenames)

#get the label files
label = open(find_phone_dir + '\\' + str(labels_file), 'r')
labels = label.read()
label_list = labels.split('\n')

#delete the last element ('')
label_list.pop()

#put the labels to a dictionary
#add file name and coordinates to a dictionary 
coords = {}

for name in label_list:
    
    name_split = name.split(' ')
    coords[name_split[0]] = tuple((float(name_split[1]), float(name_split[2])))
    
#get all the images to one vector
X = np.zeros([num_samples, 326, 490, 3])

#create vector for Y for x and y coordinates
Y = np.zeros([num_samples, 2])

for i in range(len(filenames)):
    
    im = img.imread(find_phone_dir + '\\' + filenames[i])   
    X[i, :] = im
    Y[i, 0] = coords[filenames[i]][0]
    Y[i, 1] = coords[filenames[i]][1]
    
#convert the data to black and white
def to_bandw(X, alpha):
    """
    X - data
    alpha - is the threshold to set either black or white
    """
    
    #convert the images to gray scale
    rgb_weights = [0.2989, 0.5870, 0.1140]
    
    #convert to gray scale
    X = X @ rgb_weights

    #black and white image
    X[X <= alpha] = 0 #black
    X[X > alpha] = 255 #white

    return X

#create a 32 x 48 matrix that has the pixels of the phone. This function will be used to find the prior distribution
#create the box
def box_mat(X, Y, rad = 0.05):
    """
    X - the data
    Y - labels/coordinates
    rad - is the scaling factor of the rctangle
    """
    
    #rectangle length
    x_rad = math.floor(rad * 490)
    y_rad = math.floor(rad * 326)
    
    #actual location of the phone
    x_0, y_0 = Y[0], Y[1]    
    x_0 = x_0 * 490
    y_0 = y_0 * 326
    
    
    #boundaries
    x1 = math.floor(x_0 - x_rad)
    x2 = math.floor(x_0 + x_rad)
    y1 = math.floor(y_0 - y_rad)
    y2 = math.floor(y_0 + y_rad)
    
    mat = X[y1:y2, x1:x2]
    
    return mat.astype('uint8')

#calculate the prior
def calc_prior(X, Y):
    """
    X - pixel data
    Y - coordinates
    """
    #create an empty matrix for the prior
    prior = np.zeros([32, 48])

    #collect the pictures
    for i in range(len(X)):

        X_i = X[i]
        Y_i = Y[i]

        try:

            mat = box_mat(X_i, Y_i)
            prior += mat

        except:
            
            pass
            
    #normalize m
    prior = prior/len(X)   
    
    return prior

#calculate the posterior

def calc_post(prior, like):
    
    rows = np.arange(0, prior.shape[0] * math.floor(like.shape[0]/prior.shape[0]), prior.shape[0])
    cols = np.arange(0, prior.shape[1] * math.floor(like.shape[1]/prior.shape[1]), prior.shape[1])
    
    
    
    #posterior matrix
    x_post = np.zeros([max(rows), max(cols)])

    for r in range(len(rows) - 1):

        for c in range(len(cols) - 1):

            likelihood = like[rows[r]: rows[r + 1], cols[c]: cols[c + 1]]
            post = likelihood * prior
            
            #normalized posterior
            post_n = post

            x_post[rows[r]: rows[r + 1], cols[c]: cols[c + 1]] = post_n
            
    return x_post

#train set
x_train = X
y_train = Y

#change picture to black and white
x_train_bw = to_bandw(x_train, alpha = 15)

#calculate prior
prior = calc_prior(x_train_bw, y_train)
prior = prior/255 #normalize

np.save('trained_prior', prior)

#plt.imshow(prior)
#plt.show()


# In[ ]:




