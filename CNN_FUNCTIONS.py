#import scipy.io as sio
import numpy as np
import math
import matplotlib.pyplot as plt
import numpy as np

# Fix the random state to genereate consistent results
rng = np.random.RandomState(23)

# A class for defining the structure of a CNN
class CNN(object):
    
    def __init__(self, inputSize = (32, 32, 3), layers = ['C', 'P', 'F', 'S'], convFilters = [(3, 10, 1)], downsample = [2], fcSize = [4096, 10]):
        """
        An object class for defining the CNN architecture.
        
        Inputs:
            inputSize: a tuple specifying the width, height, and channels of input picture
            layers: a list of characters ('C': convolutional, 'P': pooling, 'F': fully-connected, 'S': softmax) describing the architecture of the layers
            convFilters: a list of tuples specifying the width/height, depth, and stride of each convolutional layer filter. Zero padding is deduced from these parameters to keep the output width and height the same.
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
                filterDim, numFilters, stride = convFilters.pop(0)
                weightBound = filterDim ** 2 * self.param[-1][0][2]
                filterShape = [numFilters, self.param[-1][0][2], filterDim, filterDim]
                
                # Output size, weight filters, bias, stride, zero pad
                self.param.append([(self.param[-1][0][0], self.param[-1][0][1], numFilters), rng.normal(loc = 0.0, scale = 1.0/weightBound, size = filterShape), np.zeros([numFilters,]), stride, (filterDim - stride)/2])
                
                if not ((filterDim - stride) % 2 == 0):
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
                self.param.append([(fcOut, 1, 1), rng.normal(loc = 0.0, scale = np.sqrt(1.0 / fcIn), size = [fcOut, fcIn]), rng.normal(loc = 0.0, scale = 1.0, size = [fcOut,])])
        
        # Remove the input size
        self.param.remove(self.param[0])
        
    def feedforward(self, input):
        """
        Feedforward through the layers of the CNN, calculating the output activations for each layer.
        """
        
        # Initialize list to store NumPy arrays of output activations
        self.activations = [None] * (self.numLayers + 1)
        self.activations[0] = input
        
        # Loop forwards through layers in CNN
        for layerNum in xrange(self.numLayers):
            if self.layers[layerNum] is 'F' or 'S':
                self.fcFeedforward(layerNum)
    
    def backprop(self, t):
        """
        Backpropagate through the layers of the CNN, calculating the weight and bias gradients.
        """
        
        # Initialize list to store NumPy arrays of errors and gradients
        self.delta = [None] * self.numLayers
        self.deltaBias = [None] * self.numLayers
        self.deltaWeight = [None] * self.numLayers
        
        # Loop backwards through layers in CNN
        for layerNum in xrange(1, self.numLayers + 1):
            if self.layers[-layerNum] is 'F' or 'S':
                self.fcBackprop(t, layerNum)
                
    def fcFeedforward(self, layerNum):
        """
        Performs a fully connected layer computation and the ReLU activation
        Inputs:
            layerNum: the number of the layer in the CNN
        Outputs:
            a: output after activation function
        """
        
        # Weights and biases for current layer
        w = self.param[layerNum][1]
        b = self.param[layerNum][2]
        
        # Flatten the input activation into a 1D array (column-major order)
        a = self.activations[layerNum].flatten('F')
        
        # ReLU activation for fully-connected layers
        if self.layers[layerNum] is 'F':
            a = np.maximum(np.dot(w, a) + b, 0)
        
        # Softmax activation for final fully-connected layer
        elif self.layers[layerNum] is 'S':
            a = softmax(np.dot(w, a) + b)
        
        self.activations[layerNum + 1] = a
    
    def fcBackprop(self, t, layerNum):
        """
        Backpropagation for fully-connected layers to calculate error gradients
        Inputs:
            t: Numpy array of target (used for final softmax layer)
            layerNum: the number of the layer in the CNN
        """
        
        # Case for the final, softmax layer
        if self.layers[-layerNum] is 'S':
            # Cross-entropy error in the output
            self.delta[-layerNum] = self.activations[-layerNum] - t
            
            # Error in the bias is the error in the output of the current layer
            self.deltaBias[-layerNum] = self.delta[-layerNum]
            
            # Error in the weights is the outer product between the input activations and the output error
            self.deltaWeight[-layerNum] = np.outer(self.delta[-layerNum], self.activations[-layerNum - 1])
        
        # Case for normal fully-connected layers
        elif self.layers[-layerNum] is 'F':
            # Error in the output is the product between (1) the product between the weight matrix and the output error of the next layer and (2) the derivative of the ReLU activation of the current layer
            self.delta[-layerNum] = np.multiply(np.dot(self.param[-layerNum + 1][1].T, self.delta[-layerNum + 1]), self.activations[-layerNum] != 0)
            
            # Error in the bias is the error in the output of the current layer
            self.deltaBias[-layerNum] = self.delta[-layerNum]
            
            # Error in the weights is the outer product between the input activations and the output error
            self.deltaWeight[-layerNum] = np.outer(self.activations[-layerNum - 1].flatten('F'), self.delta[-layerNum])

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
            #reshaped_dic['data'][:,:,:,i] = np.reshape(dic['data'][i, :],(32, 32, 3))
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
    #print(np.shape(pre_layer))
    w = np.shape(pre_layer)[1]
    h = np.shape(pre_layer)[0]
    d = np.shape(pre_layer)[2]
    r = np.shape(filters)[0]
    c = np.shape(filters)[1]
    if np.shape(filters)[2] != d:
        #print('d of filter:', np.shape(filters))
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
                #print(sum(sum(sum(temp_dot_product))))
                #post_layer[j, i, k] = sum(sum(sum(temp_dot_product)))
                temp1 = temp_dot_product
                for p in range(0, d):
                    temp2 = []
                    temp2 = sum(temp1)
                    temp1 = temp2
                post_layer[j, i, k] = temp1
    return post_layer

def softmax(z):
    """Compute softmax values of each value in a 1 dimensional vector z."""
    
    # Subtract the max for numerical stability
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)