#import scipy.io as sio
import numpy as np
import math 
import matplotlib.pyplot as plt 
import numpy as np

def unpickle(file):
    '''
    this function reads the CIFAR-10 data
    each batch data contains 10000 pictures
    dict{'data','labels'}
    dict['data'] is a matrix: each row is a picture.
                              it has 3072 columns, corresponding to 32*32*3 (RGB channels)
    '''
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def conv_layer(pre_layer, filters, moving_step = 1):
    '''
    pre_layer is the input layer before doing convolution. It has dimension w*h*d (width * height * depth)
    filters has dimention r * c * d * number_of_filters(number of row * number of column * depth * number of filters)
    moving_step is the how the left corner of receptive field is moving. 
    output: post_layer with new width =  (w - c + moving_step) / moving_step
                       with new height = (h - r + moving_step) / moving_step
                       with new depth = number_of_filters
    '''
    w = np.shape(pre_layer)[1]
    h = np.shape(pre_layer)[0]
    d = np.shape(pre_layer)[2]
    r = np.shape(filters)[0]
    c = np.shape(filters)[1]
    if np.shape(filters[2]) != d:
        print('dimension dismatch! filter and input layer should have same depth.')
    number_of_filters = np.shape(filters)[3]
    new_w = int((w - c + moving_step) / moving_step)
    new_h = int((h - r + moving_step) / moving_step)
    new_d = np.shape(filters)[3]
    post_layer = np.zeros((new_w, new_h, new_d))
    
    for i in range(0, new_w):
        for j in range(0, new_h):
            for k in range(0, new_d):
                # calculate the convolution
                # first I nedd to "reverse" the filter of each depth layer,
                # But I don't think that is necessary
                temp_dot_product = np.multiply(filters[:,:,:,k], pre_layer[j*moving_step:j*moving_step + c, i*moving_step:i*moving_step + r, :])   
                print(sum(sum(sum(temp_dot_product))))
                post_layer[j, i, k] = sum(sum(sum(temp_dot_product)))
    return post_layer