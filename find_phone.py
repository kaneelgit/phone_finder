#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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
file_name = sys.argv[1]

#get current directory 
current_dir = os.getcwd()

#find phone directory
find_phone_dir = current_dir + str(file_name)

#read image
im = img.imread(find_phone_dir)   

#load the prior
prior = np.load('trained_prior.npy')


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

#turn the image to black and white with alpha = 15
x = to_bandw(im, 15)

#likelihood
like = x/255 #normalize

prior = 1 - prior
like = 1 - like

#calculate posterior distribution
posterior = calc_post(prior, like)

#estimate x and y coordinates
phone_x = np.round(np.where(posterior == np.max(posterior))[1][0]/490, 4)
phone_y = np.round(np.where(posterior == np.max(posterior))[0][0]/326, 4)

print(float(phone_x),float(phone_y))


#plt.imshow(im)
#plt.show()




         