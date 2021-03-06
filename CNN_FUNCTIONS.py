from __future__ import division
import numpy as np
from numba import jit
import threading
#import matplotlib.pyplot as plt

# Fix the random state to genereate consistent results
rng = np.random.RandomState(23)

# A class for defining the structure of a CNN
class CNN(object):
    
    def __init__(self, inputSize = (3, 32, 32), layers = ['C', 'P', 'F', 'S'], convFilters = [(3, 10, 1)], downsample = [2], fcSize = [4096, 10], parallel = True, numThreads = 4):
        """
        An object class for defining the CNN architecture.
        
        Inputs:
            inputSize: a tuple specifying the channels, width, and height of the input picture
            layers: a list of characters ('C': convolutional, 'P': pooling, 'F': fully-connected, 'S': softmax) describing the architecture of the layers
            convFilters: a list of tuples specifying the width/height, depth, and stride of each convolutional layer filter. Zero padding is deduced from these parameters to keep the output width and height the same.
            downsample: a list of ints specifying the downsampling factor of the max pooling operation for each pooling layer
            fcSize: a list of ints specifying the number of output activations for each fully connected layer and the final softmax layer
        """
        
        self.numLayers = len(layers)
        self.layers = layers
        self.parallel = parallel
        self.numThreads = numThreads
        
        self.createArchitecture(inputSize, convFilters, downsample, fcSize)
    
    def createArchitecture(self, inputSize, convFilters, downsample, fcSize):
        """
        Find the output sizes for all layers along with other relevant parameters, depending on the layer (weights, biases, stride, zero padding, pooling downsampling factor).
        """
        # Initialize list to store input/output sizes for each layer, initial weight filters/matrices in conv and FC layers, and other parameters (convolution stride and zero padding, pooling stride and filter size)
        self.param = [[inputSize, None]]
        self.deltaWeight = [None]
        self.deltaBias = [None]
        
        # Loop through the layers
        for i in xrange(self.numLayers):
            
            # Convolutional layer
            if self.layers[i] is 'C':
                filterDim, numFilters, stride = convFilters.pop(0)
                numChannels = self.param[-1][0][0]
                weightBound = filterDim ** 2 * numChannels
                filterShape = [numFilters, numChannels, filterDim, filterDim]
                
                # Output size, weight filters, bias, stride, zero pad
                self.param.append([(numFilters, self.param[-1][0][1], self.param[-1][0][2]), rng.normal(loc = 0.0, scale = 1.0/weightBound, size = filterShape), np.zeros([numFilters,]), stride, (filterDim - stride)//2])
                
                # Weight/bias error
                self.deltaWeight.append(np.zeros(filterShape))
                self.deltaBias.append(np.zeros([numFilters,]))
                
                if not ((filterDim - stride) % 2 == 0):
                    raise ValueError('Filter dimensions and stride length do not allow for zero padding to maintain width and height of input to convolational layer.')
            
            # Pooling layer
            elif self.layers[i] is 'P':
                poolSize = downsample.pop(0)
                
                # Output size, pooling size, max value index boolean mask
                self.param.append([(self.param[-1][0][0], self.param[-1][0][1] // poolSize, self.param[-1][0][2] // poolSize), poolSize, 0])
                
                # Weight/bias error
                self.deltaWeight.append(np.zeros(1))
                self.deltaBias.append(np.zeros(1))
                
                if not (self.param[-2][0][1] % poolSize == 0 and self.param[-2][0][2] % poolSize == 0):
                    raise ValueError('Pooling sizes do not divide evenly with width and height of input activation.')
            
            # FC or Softmax layer
            else:
                fcIn = self.param[-1][0][0] * self.param[-1][0][1] * self.param[-1][0][2]
                fcOut = fcSize.pop(0)
                
                # Output size, weight matrix, bias
                self.param.append([(1, 1, fcOut), rng.normal(loc = 0.0, scale = np.sqrt(1.0 / fcIn), size = [fcOut, fcIn]), rng.normal(loc = 0.0, scale = 1.0, size = [fcOut,])])
                
                # Weight/bias error
                self.deltaWeight.append(np.zeros([fcOut, fcIn]))
                self.deltaBias.append(np.zeros([fcOut,]))
        
        # Remove the input size and other initializations
        self.param.remove(self.param[0])
        self.deltaWeight.remove(self.deltaWeight[0])
        self.deltaBias.remove(self.deltaBias[0])
    
    def SGD(self, trainData, trainLabel, batchSize, epochs, eta):
        """
        Stochastic gradient descent (SGD) to train the CNN. Loop through the feedforward and backprop functions until the error converges.
        Input:
            trainData: NumPy array representing all training data, with dimension (number of training images, channels, height, width)
            trainLabel: NumPy array representing label for each training image in integer format
            batchSize: size of minibatch for SGD
            epochs: how many epochs to run
            eta: learning rate
        """
        # Total number of training examples
        numTrain = trainData.shape[0]
        
        # Total number of minibatches to train
        numBatches = numTrain//batchSize
        if numTrain % batchSize is not 0:
            raise ValueError('Batch size should divide evenly into number of training samples.')
        
        # Number of classes in the data
        numClass = np.unique(trainLabel).size
        
        # Loop through epochs
        for i in xrange(epochs):
            print('Epoch %d of %d' % (i+1, epochs))
            # Find a new random shuffling of the data for each epoch
            order = np.arange(numTrain)
            np.random.shuffle(order)
            
            # Store the indices of the shuffling
            miniBatch = np.reshape(order, (batchSize, numBatches))
            
            # Loop through each minibatch
            for j in xrange(numBatches):
                print('Minibatch %d of %d' % (j+1, numBatches))
                # Initialize arrays to accumulate errors in weights and biases
                cumWeight = [np.zeros(w.shape) for w in self.deltaWeight]
                cumBias = [np.zeros(b.shape) for b in self.deltaBias]
                
                # Loop through the samples in each minibatch
                for k in miniBatch[:, j]:
                    self.feedforward(trainData[k, :, :, :])
                    self.backprop(np.eye(numClass)[trainLabel[k]])
                    
                    # Accumulate the weight/bias errors
                    cumWeight = [nw+dnw for nw, dnw in zip(cumWeight, self.deltaWeight)]
                    cumBias = [nb+dnb for nb, dnb in zip(cumBias, self.deltaBias)]
                
                # Update weight/bias
                for l in xrange(self.numLayers):
                    if self.layers[l] is not 'P':
                        self.param[l][1] -= (eta/batchSize)*cumWeight[l]
                        self.param[l][2] -= (eta/batchSize)*cumBias[l]
    
    def feedforward(self, input):
        """
        Feedforward through the layers of the CNN, calculating the output activations for each layer.
        """
        # Initialize list to store NumPy arrays of output activations
        self.activations = [None] * (self.numLayers + 1)
        self.activations[0] = input
        
        # Loop forwards through layers in CNN
        for l in xrange(self.numLayers):
            if self.layers[l] is 'C':
                self.convFeedforward(l)
            elif self.layers[l] is 'P':
                self.poolFeedforward(l)
            else:
                self.fcFeedforward(l)
    
    def backprop(self, t):
        """
        Backpropagate through the layers of the CNN, calculating the error gradients and errors for other parameters specific to each layer. Input t is the one-hot encoded truth label for a given input image.
        """
        # Initialize list to store NumPy arrays of errors
        self.delta = [None] * (self.numLayers + 1)
        
        # Loop backwards through layers in CNN
        for l in xrange(1, self.numLayers + 1):
            if self.layers[-l] is 'S':
                # Cross entropy error at the softmax output
                self.delta[-l] = self.activations[-l] - t
                self.fcBackprop(l)
            elif self.layers[-l] is 'F':
                self.fcBackprop(l)
            elif self.layers[-l] is 'P':
                self.poolBackprop(l)
            else:
                self.convBackprop(l)
                
    def fcFeedforward(self, l):
        """
        Feedforward for fully connected layers. Compute the dot product and apply the ReLU activation function.
        """
        # Weights and biases for current layer
        w = self.param[l][1]
        b = self.param[l][2]
        
        # Flatten the input activation into a 1D array
        x = self.activations[l].flatten('C')
        
        # ReLU activation for fully-connected layers
        if self.layers[l] is 'F':
            self.activations[l + 1] = np.maximum(np.dot(w, x) + b, 0)
        
        # Softmax activation for final fully-connected layer
        elif self.layers[l] is 'S':
            self.activations[l + 1] = softmax(np.dot(w, x) + b)
    
    def fcBackprop(self, l):
        """
        Backpropagation for fully-connected layers. Calculate error gradients (delta) at the input and error in the weights/bias.
        """
        # Error in the bias is the error in the output of the current layer
        self.deltaBias[-l] = self.delta[-l]
        
        # Error in the weights is the outer product between the input activations and the output error. In this implementation, we transpose the outer product because the weight matrices are already transposed.
        self.deltaWeight[-l] = np.outer(self.delta[-l], self.activations[-l - 1].flatten('C'))
        
        # Error in the input is the element-wise product between (1) the matrix product between the weight matrix and the output error and (2) the derivative of the ReLU input activation
        self.delta[-l - 1] = np.multiply(np.dot(self.param[-l][1].T, self.delta[-l]), self.activations[-l - 1].flatten('C') != 0)
        
    def poolFeedforward(self, l):
        """
        Feedforward for the max pooling layer. Find the max pooling output and the indices for the max values.
        """
        # Find the pooling downsampling rate for the current pooling layer
        poolSize = self.param[l][1]
        
        # Find the width/height dimension of the output activation after pooling
        k = self.activations[l].shape[-1] // poolSize
        
        # Find the number of input activation channels
        depth = self.activations[l].shape[0]
        
        # Reshape the input activation in a way to find the max value of each pooling block by finding the max along certain dimensions
        actReshape = self.activations[l].reshape(depth, k, poolSize, k, poolSize)
        
        # Calculate the output activation, a.k.a. find the max values
        self.activations[l + 1] = actReshape.max(axis = (-1, -3))
        
        # Find the indices of the max values in the input activation and represent as a boolean mask. This involves finding the argmax along the rows of the reshaped activation, then finding the argmax along the columns of the max of the reshaped activation. These two argmax calculations are then one-hot encoded to form two boolean masks, and the logical and of these two masks form the boolean mask that encodes the argmax information of the original activation.
        ind1 = actReshape.argmax(axis = -1)
        mask1 = np.eye(poolSize, dtype = 'bool')[ind1].flatten('C')
        
        ind2 = actReshape.max(axis = -1).argmax(axis = -2)
        mask2 = np.eye(poolSize, dtype = 'bool')[ind2].swapaxes(-1, -2).flatten('C').repeat(poolSize)
        
        # Store the linear indices of the max values from max pooling
        self.param[l][2] = np.flatnonzero(np.logical_and(mask1, mask2))
    
    def poolBackprop(self, l):
        """
        Backpropagation for max pooling layer. Propagate the errors from the output to the indices in the input that correspond to the max values.
        """
        # Initialize array for error at the input with all zeros
        self.delta[-l - 1] = np.zeros(self.activations[-l - 1].size)
        
        # Assign errors at the output to the indices of the max values for the current max pooling layer
        self.delta[-l - 1][self.param[-l][2]] = self.delta[-l].flatten('C')
        
        # Reshape the input error array to the same shape as the input activation
        self.delta[-l - 1] = np.reshape(self.delta[-l - 1], self.activations[-l - 1].shape)
    
    def convFeedforward(self, l):
        """
        Feedforward for convolutional layer. Compute the convolution between the input activation and the filters and apply the ReLU activation function.
        """
        # Parameters: input activation, filter, bias, stride, zero pad width
        x = self.activations[l]
        w = self.param[l][1]
        b = self.param[l][2][:, np.newaxis, np.newaxis]
        stride = self.param[l][3]
        zeroPad = self.param[l][4]
        
        # Do a convolution plus bias and apply the ReLU activation function
        self.activations[l + 1] = np.maximum(conv_layer(x, w, stride, zeroPad, parallel = self.parallel, numThreads = self.numThreads) + b, 0)
    
    def convBackprop(self, l):
        """
        Backpropagation for convolutional layer. Calculate the errors at the input and the convolutional filter weights.
        """
        # Parameters: transposed filter weights, stride, zero pad width
        w = self.param[-l][1].swapaxes(0, 1)
        stride = self.param[-l][3]
        zeroPad = self.param[-l][4]
        
        # Cross-correlate the output error with the transposed weights and multiply by the derivative of the ReLU input activation to get the input error
        self.delta[-l - 1] = np.multiply(conv_layer(self.delta[-l], w, stride, zeroPad, flipFilter = False, parallel = self.parallel, numThreads = self.numThreads), self.activations[-l - 1] != 0)
        
        # Sum values in each activation map to get the error in the bias
        self.deltaBias[-l] = np.sum(self.delta[-l], axis = (1, 2))
        
        # Cross-correlate the input activation with the output error to get the error in the weights
        self.deltaWeight[-l] = conv_layer(self.activations[-l - 1], self.delta[-l], stride, zeroPad, flipFilter = False, dW = True, parallel = self.parallel, numThreads = self.numThreads)

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

def conv_layer(pre_layer, filters, stride, zeroPad, flipFilter = True, dW = False, parallel = True, numThreads = 1):
    '''
    Do convolution on the pre_layer, and generate the post_layer
    Inputs:
        pre_layer: input activation with dimension d * h * w (depth * height * width)
        filters: dimension numFilters * filterDepth * r * c (number of filters * depth * number of rows * number of columns)
        stride: how much the left corner of receptive field is moving
        zeroPad: how many layers of zeros to pad the input activations
        flipFilter: 1 to flip filter (convolution), 0 to not flip (correlation)
        dW: 1 to perform a different method for the backpropagation of the error in the weights, where the inputs are the input activation and the output error, and the output is the set of errors for each filter in the layer
    Output:
        post_layer: output activation
    '''
    # Find depth, height, and width of input activation
    d, h, w = pre_layer.shape
    
    if dW is False:
        # Find number of filters, filter depth, filter height, and filter width
        numFilters, filterDepth, r, c = filters.shape
    else:
        # Find the depth, heigh, and width of output error
        numFilters, r, c = filters.shape
    
    # Check to see if filter depth and input activation depth are the same
    if dW is False and filterDepth is not d:
        raise ValueError('Dimension mismatch! Filter and input activation should have the same depth.')
    
    # Zero-pad the input along the last two axes
    padWidth = ((0,0), (zeroPad, zeroPad), (zeroPad, zeroPad))
    pre_layer = np.lib.pad(pre_layer, padWidth, 'constant')
    
    # For a convolution, flip the filters
    if flipFilter is True and dW is False:
#        filters = np.rot90(filters, k = 2, axes = (-1, -2))
        # For NumPy version < 1.12
        filters = np.rot90(filters.transpose((2, 3, 0, 1)), k = 2).transpose((2, 3, 0, 1))
    
    # Parallel convolution for forward propagation
    if dW is False and parallel is True:
        post_layer = np.zeros((numFilters, h, w))
        
        # Reset number of threads if greater than height of input activation
        if numThreads > h:
            numThreads = h
            
        # Initialize chunks for each thread
        chunk_size = int(np.ceil(h/numThreads))
        i_list = np.linspace(0, h-1, num=h)
        i_list = i_list.astype(int)
        i_chunks = [None]*numThreads
        
        for p in xrange(numThreads):
            if p != numThreads - 1:
                i_chunks[p] = i_list[p*chunk_size:(p+1)*chunk_size]
            else:
                i_chunks[p] = i_list[p*chunk_size:h]
        
        # Process each chunk in parallel through the para_conv Numba function
        threads = [None]*numThreads
        for p in xrange(numThreads):
            chunk_size_new = len(i_chunks[p])
            thread = threading.Thread(target = para_conv, args = (post_layer, chunk_size_new, i_chunks[p], filters, pre_layer, stride, r, c, numFilters, w))
            threads[p] = thread
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    
    # Parallel convolution for backpropating error wrt weights
    elif dW is True and parallel is True:
        post_layer = np.zeros((numFilters, d, (h - r + 2*zeroPad)//stride + 1, (w - c + 2*zeroPad)//stride + 1))
        
        # Reset number of threads if greater than height of input activation
        if numThreads > post_layer.shape[2]:
            numThreads = post_layer.shape[2]
        
        # Initialize chunks for each thread
        chunk_size = int(np.ceil(post_layer.shape[2]/numThreads))
        i_list = np.linspace(0, post_layer.shape[2] - 1, num = post_layer.shape[2])
        i_list = i_list.astype(int)
        i_chunks = [None]*numThreads
        
        for p in xrange(numThreads):
            if p != numThreads - 1:
                i_chunks[p] = i_list[p*chunk_size:(p+1)*chunk_size]
            else:
                i_chunks[p] = i_list[p*chunk_size:post_layer.shape[2]]
        
        # Process each chunk in parallel through the para_conv_v2 Numba function
        threads = [None]*numThreads
        for p in xrange(numThreads):
            chunk_size_new = len(i_chunks[p])
            thread = threading.Thread(target = para_conv_v2, args = (post_layer, chunk_size_new, i_chunks[p], filters, pre_layer, stride, r, c, d, numFilters))
            threads[p] = thread
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    
    # Nonparallel convolution for forward propagation
    elif dW is False and parallel is False:
        post_layer = np.zeros((numFilters, h, w))
        single_conv(post_layer, filters, pre_layer, h, w, numFilters, stride, r, c)
    
    # Nonparallel convolution for backpropating error wrt weights
    elif dW is True and parallel is False:
        h = (h - r + 2*zeroPad)//stride + 1
        w = (w - c + 2*zeroPad)//stride + 1
        post_layer = np.zeros((numFilters, d, h, w))
        single_conv_v2(post_layer, filters, pre_layer, d, h, w, numFilters, stride, r, c)
        
    return post_layer

@jit(nopython = True, nogil = True)
def para_conv(post_layer, chunk_size, i_list, filters, pre_layer, stride, r, c, numFilters, w):
    '''
    processed by one block
    '''
    for i in i_list:
        for j in xrange(w):
            for k in xrange(numFilters):
                # Multiply element-wise
                temp_dot_product = np.multiply(filters[k,:,:,:], pre_layer[:, i*stride:i*stride + r, j*stride:j*stride + c])
                    
                # Sum up element-wise multiplication to complete dot product
                post_layer[k, i, j] = np.sum(temp_dot_product)
                
@jit(nopython = True, nogil = True)
def para_conv_v2(post_layer, chunk_size, i_list, filters, pre_layer, stride, r, c, d, numFilters):
    '''
    processed by one block
    '''
    for i in i_list:
        for j in xrange(post_layer.shape[3]):
            for k in xrange(d):
                for p in xrange(numFilters):
                    # Multiply element-wise
                    temp_dot_product = np.multiply(filters[p,:,:], pre_layer[k, i*stride:i*stride + r, j*stride:j*stride + c])
                    
                    # Sum up element-wise multiplication to complete dot product
                    post_layer[p, k, i, j] = np.sum(temp_dot_product)

# @jit(nopython = True, nogil = True)
# def para_conv(post_layer, chunk_size, i_list, j_list, k_list, filters, pre_layer, stride, r, c):
#     '''
#     processed by one block
#     '''
#     for i in xrange(chunk_size):
#         temp_dot_product = np.multiply(filters[k_list[i],:,:,:], pre_layer[:, i_list[i]*stride:i_list[i]*stride + r, j_list[i]*stride:j_list[i]*stride + c])
#         # Sum up element-wise multiplication to complete dot product
#         post_layer[k_list[i], i_list[i], j_list[i]] = np.sum(temp_dot_product)

@jit(nopython = True, nogil = True)
def single_conv(post_layer, filters, pre_layer, h, w, numFilters, stride, r, c):
    for i in xrange(h):
            for j in xrange(w):
                for k in xrange(numFilters):
                    # Multiply element-wise
                    temp_dot_product = np.multiply(filters[k,:,:,:], pre_layer[:, i*stride:i*stride + r, j*stride:j*stride + c])
                    
                    # Sum up element-wise multiplication to complete dot product
                    post_layer[k, i, j] = np.sum(temp_dot_product)
                    
@jit(nopython = True, nogil = True)
def single_conv_v2(post_layer, filters, pre_layer, d, h, w, numFilters, stride, r, c):
    for i in xrange(h):
        for j in xrange(w):
            for k in xrange(d):
                for p in xrange(numFilters):
                    # Multiply element-wise
                    temp_dot_product = np.multiply(filters[p,:,:], pre_layer[k, i*stride:i*stride + r, j*stride:j*stride + c])
                    
                    # Sum up element-wise multiplication to complete dot product
                    post_layer[p, k, i, j] = np.sum(temp_dot_product)

def softmax(z):
    """Compute softmax values of each value in a 1 dimensional vector z."""
    
    # Subtract the max for numerical stability
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)
# Some test code
#train = unpickle('data_batch_1')
#trainData = train['data'].reshape((10000, 3, 32, 32))
#trainLabel = train['labels']
#x = CNN(inputSize = (3, 32, 32), layers = ['C', 'P', 'F', 'S'], convFilters = [(3, 3, 1)], downsample = [2], fcSize = [256, 10])
#x.SGD(trainData[0: 1000, :, :, :], trainLabel[0: 1000], 100, 1, 0.9)