#import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np

# Fix the random state to genereate consistent results
rng = np.random.RandomState(23)

# A class for defining the structure of a CNN
class CNN(object):
    
    def __init__(self, inputSize = (32, 32, 3), layers = ['C', 'P', 'F', 'S'], convFilters = [(3, 3, 10, 1)], downsample = [2], fcSize = [4096, 10]):
        """
        An object class for defining the CNN architecture.
        
        Inputs:
            inputSize: a tuple specifying the width, height, and channels of input picture
            layers: a list of characters ('C': convolutional, 'P': pooling, 'F': fully-connected, 'S': softmax) describing the architecture of the layers
            convFilters: a list of tuples specifying the width, height, depth, and stride of each convolutional layer filter. Zero padding is deduced from these parameters to keep the output width and height the same.
            downsample: a list of ints specifying the downsampling factor of the max pooling operation for each pooling layer
            fcSize: a list of ints specifying the number of output activations for each fully connected layer and the final softmax layer
        """
        self.numLayers = len(layers)
        self.numConvLayers = layers.count('C')
        self.numPoolLayers = layers.count('P')
        self.numFCLayers = layers.count('F') + layers.count('S')
        
        self.layers = layers
        
        self.createArchitecture(inputSize, convFilters, downsample, fcSize)
    
    def createArchitecture(self, inputSize, convFilters, downsample, fcSize):
        """
        Find the output sizes for all layers along with other relevant parameters, depending on the layer (weights, biases, stride, zero padding, pooling downsampling factor).
        """
        
        # Initialize list to store input/output sizes for each layer, initial weight filters/matrices in conv and FC layers, and other parameters (convolution stride and zero padding, pooling stride and filter size)
        self.param = [[inputSize, None]]
        
        # Loop through the layers
        for i in xrange(self.numLayers):
            
            # Convolutional layer
            if self.layers[i] is 'C':
                filterXDim, filterYDim, numFilters, stride = convFilters.pop(0)
                weightBound = filterXDim * filterYDim * self.param[-1][0][2]
                filterShape = [filterXDim, filterYDim, self.param[-1][0][2], numFilters]
                
                # Output size, weight filters, bias, stride, zero pad
                self.param.append([(self.param[-1][0][0], self.param[-1][0][1], numFilters), rng.normal(loc = 0.0, scale = 1.0/weightBound, size = filterShape), np.zeros([numFilters,]), stride, (filterXDim - stride)/2])
                
                if not ((filterXDim - stride) % 2 == 0 and (filterYDim - stride) % 2 == 0):
                    raise ValueError('Filter dimensions and stride length do not allow for zero padding to maintain width and height of input to convoluational layer.')
            
            # Pooling layer
            elif self.layers[i] is 'P':
                poolSize = downsample.pop(0)
                
                # Output size, pooling size
                self.param.append([(self.param[-1][0][0] / poolSize, self.param[-1][0][1] / poolSize, self.param[-1][0][2]), poolSize])
                
                if not (self.param[-1][0][0] % poolSize == 0 and self.param[-1][0][1] % poolSize == 0):
                    raise ValueError('Pooling sizes do not divide evenly with width and height of input activation.')
            
            # FC or Softmax layer
            elif self.layers[i] is 'F' or 'S':
                fcIn = self.param[-1][0][0] * self.param[-1][0][1] * self.param[-1][0][2]
                fcOut = fcSize.pop(0)
                
                # Output size, weight matrix, bias
                self.param.append([(fcOut, 1, 1), rng.normal(loc = 0.0, scale = np.sqrt(1.0 / fcOut), size = [fcIn, fcOut]), rng.normal(loc = 0.0, scale = 1.0, size = [fcOut,])])
        
        # Remove the input size
        self.param.remove(self.param[0])
        

def unpickle(file):
    '''
    Purpose: --- this function reads the CIFAR-10 data
    each batch data contains 10000 pictures
    dict{'data','labels'}
    dict['data'] is a matrix: each row is a picture.
                              it has 3072 columns, corresponding to 32*32*3 (RGB channels)
    '''
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def reshape_batch_data(dic):
    '''
    Purpose: --- reshape each picture data from a vector to a 32*32*3 pictures
    Input: --- dic is the result from unpickle function
    output: --- reshaped_dic is a dictionary, with 'data' and 'labels'
                reshaped_dic['data'] is an array with dimension 32*32*3*10000
    '''
    number_of_pictures = np.shape(dic['labels'])[0]
    reshaped_dic = {}
    # copy the labels from dic to reshaped_dic
    reshaped_dic['labels'] = dic['labels']
    # reshape the data
    reshaped_dic['data'] = np.zeros((32, 32, 3, 10000))
    for i in range(0, number_of_pictures):
        for j in range(0,3):
            reshaped_dic['data'][:,:,j,i] = np.reshape(dic['data'][i,j*1024:(j+1)*1024], (32,32))
    return reshaped_dic
    

def conv_layer(pre_layer, filters, moving_step = 1):
    '''
    Purpose: --- conv_layer is the function doing convolution on the pre_layer, and generate the post_layer
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
        print('dimension mismatch! filter and input layer should have same depth.')
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
                #post_layer[j, i, k] = sum(sum(sum(temp_dot_product)))
                temp1 = temp_dot_product
                for p in range(0, d):
                    temp2 = []
                    temp2 = sum(temp1)
                    temp1 = temp2
                post_layer[j, i, k] = temp1
    return post_layer

def fc_layer(x, numOutput, w = None):
    """
    Performs a fully connected layer computation and the ReLU activation
    Inputs:
        x: NumPy array input
        numOutput: number of output neurons
        w: weight matrix
    Outputs:
        a: output after activation function
        w: weight matrix (for backpropagation)
    """
    # Flatten the input into a 1D array (column-major order)
    x = x.flatten('F')
    
    # Add the bias term
    x = np.r_[x, 1]
    
    # Check if there is already an input weight matrix before initializing a new one, which is normally distributed with zero mean and unit variance
    if w is None:
        w = np.random.randn(x.shape[0], numOutput)
    
    # Perform a fully-connected computation
    z = np.dot(w.T, x)
    
    # Run the output through the ReLU activation function
    a = np.maximum(z, 0)
    
    return a, w