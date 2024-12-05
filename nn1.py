
"""
Building Neural Networks from Scratch 

"""
#%%

# coding a neuron
# we input and weights in a list
# take 3 inputs

inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2

outputs = (inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias)

print(outputs) 

# Weights are indicative of importance for an input
# higher the weight, important the attribute

#%%
# coding with 4 inputs now... 
# when 4 inputs , there should be 4 weights , but bias does not increase
# with the number of inputs


inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
output = (inputs[0]*weights[0] +
 inputs[1]*weights[1] +
 inputs[2]*weights[2] +
 inputs[3]*weights[3] + bias)

print(output)

#%%
# coding layer of neurons

# input values : x1, x2, x3 and x4
# weight values : 4 weight values for each neuron
# bias term for each neuron


inputs = [1, 2, 3, 2.5]
#      = [x1, x2, x3, x4]


# list of lists
# each list shows weight of a respective neuron 
weights = [[0.2, 0.8, -0.5, 1], # 4 weights to 1st neuron
           [0.5, -0.91, 0.26, -0.5], # 4 weights to 2nd neuron
           [-0.26, -0.27, 0.17, 0.87]] # 4 weights to 3rd neuron

#1st neuron
weights1 = weights[0]
print(weights1) #LIST OF WEIGHTS ASSOCIATED WITH 1ST NEURON : 
# W11, W12, W13, W14

#2nd neuron
weights2 = weights[1] 
print(weights2)#LIST OF WEIGHTS ASSOCIATED WITH 2ND NEURON : 
# W21, W22, W23, W24

# 3rd neuron
weights3 = weights[2] #LIST OF WEIGHTS ASSOCIATED WITH 3RD NEURON : 
# W31, W32, W33, W34
print(weights3)


# bias term for each neuron
biases = [2, 3, 0.5]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [
 # Neuron 1:
 inputs[0]*weights1[0] +
 inputs[1]*weights1[1] +
 inputs[2]*weights1[2] +
 inputs[3]*weights1[3] + bias1,
 # Neuron 2:
 inputs[0]*weights2[0] +
 inputs[1]*weights2[1] +
 inputs[2]*weights2[2] +
 inputs[3]*weights2[3] + bias2,
 # Neuron 3:
 inputs[0]*weights3[0] +
 inputs[1]*weights3[1] +
 inputs[2]*weights3[2] +
 inputs[3]*weights3[3] + bias3]

print(outputs)

# [4.8, 1.21, 2.385]
# output of neuron [1,2,3]


#%%

# what if there are 50 neurons , we need to loop it .

inputs = [1, 2, 3, 2.5]

##LIST OF WEIGHTS
weights = [[0.2, 0.8, -0.5, 1],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]

##LIST OF BIASES
biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
 # Zeroed output of given neuron
 neuron_output = 0
 # For each input and weight to the neuron
 for n_input, weight in zip(inputs, neuron_weights):
 # Multiply this input by associated weight
 # and add to the neuron's output variable
   neuron_output += n_input*weight ## W31*X1 + W32*X2 + W33*X3 + W34*X4
   # Add bias
 neuron_output += neuron_bias ## ## W31*X1 + W32*X2 + W33*X3 + W34*X4 + B3
 # Put neuron's result to the layer's output list
 layer_outputs.append(neuron_output)
print(layer_outputs)     

#%%

# Single Neuron using numpy
import numpy as np

# np.dot(X,W) + Bias

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
outputs = np.dot(weights, inputs) + bias
print(outputs)


#%%
# Coding layer using Numpy

inputs = [1.0, 2.0, 3.0, 2.5] #x1,x2,x3,x4

# 3 neurons

weights = [[0.2, 0.8, -0.5, 1], # w11,w12,w13,w14 ... 
           [0.5, -0.91, 0.26, -0.5], # w21,w22,w23,w24 ...
           [-0.26, -0.27, 0.17, 0.87]] #w31,w32,w33,w34

biases = [2.0, 3.0, 0.5] 

# A dot product of a matrix and a vector results in a list of dot products. 
#The np.dot() method treats the matrix as a list of vectors and performs a dot product of each of those vectors with the other vector

layer_outputs = np.dot(weights, inputs) + biases

# if you look closely, it is a vector and matrix multiplication
# biases vector and weights matrix

# np.dot(W,x) ... what will this be ? 
# now here you can see matrix comes first, this case is np.dot(b,a)


# w11,w12,w13,w14 : W  (w1)
# w21,w22,w23,w24 : W  (w2)
# w31,w32,w33,w34 : W  (w3)

# [x1 x2 x3 x4] : X 

# this can be expressed as np.dot(W,X) + B 
# resultant dimension : (3,4) and (4,1) - hence (3,1)

print(layer_outputs)

# if you perform np.dot(X,W) : it will give wrong answer
# because the values will not correspond to their neurons

#%%


inputs = [[1.0, 2.0, 3.0, 2.5], # batch 1
          [2.0, 5.0, -1.0, 2.0], # batch 2
          [-1.5, 2.7, 3.3, -0.8]] # batch 3

weights = [[0.2, 0.8, -0.5, 1], # weight neuron 1
 [0.5, -0.91, 0.26, -0.5], # weight neuron 2
 [-0.26, -0.27, 0.17, 0.87]] # weight neuron 3

biases = [2.0, 3.0, 0.5]

# NOTE: WE CAN'T TRANSPOSE LISTS IN PYTHON, SO WE HAVE THE CONVERT THE WEIGHTS MATRIX INTO AN ARRAY FIRST
outputs = np.dot(inputs, np.array(weights).T) + biases 
print(outputs)

# we added np.array : we cannot transpose of lists in python 
# weights are defined here as list of lists
# hence we need to convert them to array and then take transpose

# 1st row : 1st batch output
# 2nd row : 2nd batch output
# 3rd row : 3rd batch output

#%%

# For 2 layers and batch of data using numpy

import numpy as np

inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1],  # first weight matrix : 3X4 
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]    # 1st bias matrix : (3X1) 
 
weights2 = [[0.1, -0.14, 0.5],  # second weight matrix : 3X3 
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]  # 2nd bias matrix : (3X1)

# Convert lists to numpy arrays
inputs_array = np.array(inputs)
weights_array = np.array(weights)
biases_array = np.array(biases)
weights2_array = np.array(weights2)
biases2_array = np.array(biases2)

# Calculate the output of the first layer
layer1_outputs = np.dot(inputs_array, weights_array.T) + biases_array
print(layer1_outputs)
# Calculate the output of the second layer
# here we can see the layer_1 output is taken as input in the 2nd layer
layer2_outputs = np.dot(layer1_outputs, weights2_array.T) + biases2_array
print(layer2_outputs)

#%%

# Exercise try create 4 hidden layer network and check the output

#%%

# Implementing a Dense Layer class

# nnfs 
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
X, y = spiral_data(samples=100, classes=3) 
# first class : blue, 2nd : green , 3rd : red
plt.scatter(X[:, 0], X[:, 1])
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()

# currently the neural network is unaware of the difference in colours
# because we have not encoded the data 

# Dense layer
class Layer_Dense:
 # Layer initialization 
 """
 n_inputs = 2, n_neurons = 3
 
 """
 def __init__(self, n_inputs, n_neurons): 
 # Initialize weights and biases
   self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
   # np.random.randn(n_inputs, n_neurons)
   # assigning weights randomly at the moment
   # 0.01 : to prevent weights from becoming very large
   self.biases = np.zeros((1, n_neurons))  # as of now , bias = 0                            

 # Forward pass
# here no transpose here, only np.dot(x,w) + bias
 def forward(self, inputs):
 # Calculate output values from inputs, weights and biases
   self.output = np.dot(inputs, self.weights) + self.biases

# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input and 3 neurons 
dense1 = Layer_Dense(2, 3)
# Perform a forward pass of our training data through this layer
dense1.forward(X) # X.W + B

# every single input will have 3 outputs
# 100 inputs : 100 rows and 3 cols : this will be the output

# Let's see output of the first few samples:
print(dense1.output[:5]) #plotting first 5 out of 100
# every row represents the output of the 3 neurons

# for example : [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
#             : [output : neuron1 , output :neuron2 , output:neuron 3]
 


