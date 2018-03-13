from __future__ import print_function
import os
from tkinter import * 
from tkinter import ttk
from tkinter.filedialog import askdirectory
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from os import listdir
from os.path import isfile, isdir, join
import scipy
import PIL
from PIL import Image

#卷積層
def conv2d(x, W, b, strides=1):
	x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

#池化層
def maxpool2d(x, k=2):
	return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
						  padding='SAME')


# Create model
def conv_net(x, weights, biases,dropout):

	x = tf.reshape(x, shape=[-1, 28, 28, 1])


	conv1 = conv2d(x, weights['wc1'], biases['bc1'])
	conv1 = maxpool2d(conv1, k=2)


	conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
	conv2 = maxpool2d(conv2, k=2)

	# 完全連接層(先reshape feature maps)
	fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
	fc1 = tf.nn.relu(fc1)
	#Dropout
	fc1 = tf.nn.dropout(fc1, dropout)
	# Output
	out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
	return out
	
