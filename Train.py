from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

import matplotlib
import scipy.io as sio
import cPickle as pickle
import numpy as np
import math

image_width = 72
image_height= 54
pixel_depth = 255

with open('dataset.pickle','rb') as f:
    save = pickle.load(f)
    dataset_label = save['label']
    dataset_img = save['img']

train_label=dataset_label[0:200000]
train_img= dataset_img[0:200000,:]


valid_label=dataset_label[200001:210000]
valid_img=dataset_img[200001:210000,:]


test_label=dataset_label[210001:213658]
valid_img=dataset_img[210001:213658,:]

n_sample = len(train_img)

x = tf.placeholder(tf.float32, shape=[None,image_width*image_height])
y_ = tf.placeholder(tf.float32, shape=[None, 1])



x = tf.placeholder(tf.float32,[None,2160]);
W = tf.Variable(tf.zeros([2160,1]));