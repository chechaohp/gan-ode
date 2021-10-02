##################################################
## Project: RotNIST
## Script purpose: To download MNIST dataset and append new rotated digits to it
## Date: 21st April 2018
## Author: Chaitanya Baweja, Imperial College London
##################################################

### modified from the original source to use with pytorch###


# usage
# python3 main.py num_angles num_seq

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gzip
import os
import sys
import numpy as np
from torchvision import datasets
import time
import csv
from scipy import ndimage
from scipy import io
from six.moves import urllib
from PIL import Image
# from imageio import imwrite
from pathlib import Path
#Url for downloading MNIST dataset
URL = 'http://yann.lecun.com/exdb/mnist/'
#Data Directory where all data is saved
DATA_DIRECTORY = "data"

# Params for MNIST


'''
Download the data from Yann's website, unless it's already here.
filename: filepath to images
Returns path to file
'''
def download(filename):
    #Check if directory exists
    if not os.path.exists(DATA_DIRECTORY):
        os.mkdir(DATA_DIRECTORY)
    filepath = os.path.join(DATA_DIRECTORY, filename)
    #Check if file exists, if not download
    if not os.path.exists(filepath):
        filepath, _ = urllib.request.urlretrieve(URL + filename, filepath)
        size = os.path.getsize(filepath)
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath
'''
Extract images from given file path into a 4D tensor [image index, y, x, channels].
Values are rescaled from [0, 255] down to [-0.5, 0.5].
filename: filepath to images
num: number of images
60000 in case of training
10000 in case of testing
Returns numpy vector
'''
def extract_data(filename, num):
    print('Extracting', filename)
    #unzip data
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num * 1)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = (data - (255 / 2.0)) / 255 #rescaling value to [-0.5,0.5]
        data = data.reshape(num, 28, 28, 1) #reshape into tensor
        data = np.reshape(data, [num, -1])
    return data

'''
Extract the labels into a vector of int64 label IDs.
filename: filepath to labels
num: number of labels
60000 in case of training
10000 in case of testing
Returns numpy vector
'''
def extract_labels(filename, num):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
        num_labels_data = len(labels)
        one_hot_encoding = np.zeros((num_labels_data,10))
        one_hot_encoding[np.arange(num_labels_data),labels] = 1
        one_hot_encoding = np.reshape(one_hot_encoding, [-1, 10])
    return one_hot_encoding

'''
Augment training data with rotated digits
images: training images
labels: training labels
'''
def rotate_data(images, labels, K=16):

    # angles = np.linspace(0,359.99,K)
    # angles = np.linspace(0,360*(K-1)/K,K)
    X = np.zeros([len(images),K,784])
    Y = np.zeros([len(images),1])

    directory = os.path.dirname("data/rotated")
    if not os.path.exists("data/rotated"):
        os.mkdir("data/rotated")

    k = 0 # counter
    for x, y in zip(images, labels):
        # random angle differently for each image
        end_angle = np.random.uniform(-269.99,269.99)
        if end_angle < 0:
            end_angle -= 90
        else:
            end_angle += 90
        angles = np.linspace(0,end_angle,K)
        #
        Y[k,0] = np.where(y==1)[0][0]
        bg_value = -0.5 # this is regarded as background's value black

        image = np.reshape(x, (-1, 28))
        for i in range(K):
            # rotate the image with random degree
            angle = angles[i]
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            # shift = np.random.randint(-2, 2, 2)
            # new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            #code for saving some of these for visualization purpose only
            if k<10:
            	tmp = (new_img*255) + (255 / 2.0)
            	im = Image.fromarray(tmp)
            	NAME = DATA_DIRECTORY+"/rotated/"+str(k)+"-"+str(i)+".jpeg"
            	im.convert('RGB').save(NAME)
            X[k,i,:] = np.reshape(new_img, 784)

        k = k+1
        if k%100==0:
            print ('expanding data : %03d / %03d' % (k,np.size(images,0)))

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    X = np.array(X)
    Y = np.array(Y)
    X = X / (np.max(X,2,keepdims=True) - np.min(X,2,keepdims=True))
    X = X - np.min(X,2,keepdims=True)
    # X = X + 0.5
    # X = X / (np.max(X)-np.min(X))
    rnd_idx = np.random.shuffle(np.arange(0,X.shape[0]))
    print(X.shape)

    return X[rnd_idx,:], Y[rnd_idx,:]

'''
Main function to prepare the entire data
'''
def prepare_MNIST_data():
    # Get the data.
    train_data_filename = download('train-images-idx3-ubyte.gz')
    train_labels_filename = download('train-labels-idx1-ubyte.gz')
    test_data_filename = download('t10k-images-idx3-ubyte.gz')
    test_labels_filename = download('t10k-labels-idx1-ubyte.gz')
    # train_data_filename = 'data/train-images-idx3-ubyte.gz'
    # train_labels_filename = 'data/train-labels-idx1-ubyte.gz'
    # test_data_filename = 'data/t10k-images-idx3-ubyte.gz'
    # test_labels_filename = 'data/t10k-labels-idx1-ubyte.gz'


    K = 16
    tr_size = 10000


    valid_size = int(tr_size/10)
    test_size  = int(tr_size/10)

    # Extract it into numpy arrays.
    train_data   = extract_data(train_data_filename, tr_size+valid_size)
    train_labels = extract_labels(train_labels_filename, tr_size+valid_size)
    valid_data   = train_data[:valid_size, :]
    valid_labels = train_labels[:valid_size,:]
    train_data   = train_data[valid_size:, :]
    train_labels = train_labels[valid_size:,:]

    # get only threes
    # idx = np.where(train_labels[:,3]==1)
    # train_data = train_data[idx]
    # train_labels = train_labels[idx]

    test_data   = extract_data(test_data_filename, test_size)
    test_labels = extract_labels(test_labels_filename, test_size)

    rot_train_data, rot_train_labels = rotate_data(train_data, train_labels, K)
    rot_valid_data, rot_valid_labels = rotate_data(valid_data, valid_labels, K)
    rot_test_data,  rot_test_labels  = rotate_data(test_data,  test_labels, K)

    return rot_train_data, rot_valid_data, rot_test_data, rot_train_labels, rot_valid_labels, rot_test_labels

rot_train_data, rot_valid_data, rot_test_data, rot_train_labels, rot_valid_labels, rot_test_labels = prepare_MNIST_data()
io.savemat('drive/MyDrive/MNIST/rot-mnist_rand.mat',{'X':rot_train_data,'Y':rot_train_labels})