"""
Activation Functions

"""
#%%

#%% 

"Relu function "
# relu : np.maximum(0,x)
import numpy as np
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = np.maximum(0, inputs)
print(output)

#%%


class Activation_ReLU:
 # Forward pass
 def forward(self, inputs):
 # Calculate output values from input
  self.output = np.maximum(0, inputs)
  
  
#%%

# nnfs 
from nnfs.datasets import spiral_data
import numpy as np
import nnfs
nnfs.init()
import matplotlib.pyplot as plt
X, y = spiral_data(samples=100, classes=3) 
# creates 3 classes,  each containing 100 samples
# first class : blue, 2nd : green , 3rd : red
plt.scatter(X[:, 0], X[:, 1])
plt.show()

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
print(dense1)

# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Make a forward pass of our training data through this layer
dense1.forward(X)
# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)
# Let's see output of the first few samples:
print(activation1.output[:5])


"Final activation layer cannot be Relu especially in classification tasks"

#%%
# some exercises

A = [[1, 2, 3], [4, 5, 6], [7, 8,9]]
print(np.sum(A))

print(np.sum(A, axis = 0))
print(np.sum(A, axis = 0).shape)

print(np.sum(A, axis = 1))
print(np.sum(A, axis = 1).shape)

print(np.sum(A, axis = 0,keepdims = True)) # keepdims retains all the dimensions
print(np.sum(A, axis = 0,keepdims = True).shape)

print(np.sum(A, axis = 1,keepdims = True))
print(np.sum(A, axis = 1,keepdims = True).shape)

print(np.max(A, axis = 0))
print(np.max(A, axis = 1))
 
#%%

"How activation function for softmax is implemented ? "

# imagine this like a matrix

inputs = [[1, 2, 3, 2.5],
          [2., 5., -1., 2],
          [-1.5, 2.7, 3.3, -0.8]]

# subtract maximum value from each row input
# take max value of the first row subtract it all from all elements of 1st row
# this is to avoid exponential taking larger values

# Get unnormalized probabilities : take exponential values
exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
# if you put axis = 0, it will take max value from 1st column

# Normalize them for each sample
# take exponentiated values divide by sum
probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
print(probabilities)
# first row corresponds to first batch , if you sum them it will sum to 1
# [0.06414769 0.17437149 0.47399085 0.28748998]
# [0.04517666 0.90739747 0.00224921 0.04517666]
#  [0.00522984 0.34875873 0.63547983 0.0105316 ]
np.sum(probabilities, axis = 1)


#%%
# Softmax activation class : gives probabilities in the output
class Activation_Softmax:
 # Forward pass
 def forward(self, inputs):
 # Get unnormalized probabilities
  exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
 # Normalize them for each sample
  probabilities = exp_values / np.sum(exp_values, axis=1,keepdims=True)
  self.output = probabilities
  
#%%

"Application of ReLu and Softmax in forward pass"

X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 neurons
dense1 = Layer_Dense(2, 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU() #after layer 1 we use relu
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Layer_Dense(3, 3)
# Create Softmax activation (to be used with Dense layer):
activation2 = Activation_Softmax() #after layer 2 we use softmax 

# Make a forward pass of our training data through this layer
# this will generate output from 1st layer : wx+b
dense1.forward(X)
# the output of dense 1 will be then passed to the 1st activation - ReLU

# Make a forward pass through activation function
# it takes the output of first dense layer here
"output of the 1st layer"
activation1.forward(dense1.output)
# Make a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
"we use output of the previous activation as the input"
dense2.forward(activation1.output)
# Make a forward pass through activation function
# it takes the output of second dense layer here
"then finally we apply softmax on the layer 2 output"
activation2.forward(dense2.output)
# Let's see output of the first few samples:
print(activation2.output[:5]) # 3 columns since 3 output neurons
# here too, each batch will sum up to 1
# [prob of being red, prob of being green, prob of being blue]


#%%

"FORWARD PASS WITHOUT LOSS FUNCTION"


"""
Structure:

input x1 : feeds into the weight matrix (w1)
output: np.dot(x.w1+b1)
then output is being into Activation Function : ReLU
the output from activation function is f1=ReLU(x.w1+b1)
this output(f1) is then fed into the weight matrix of the 2nd layer
and the output from this will be f1*w2+b2
then this output (f1*w2+b2) will be fed into the Activation Function : Softmax
Final Output : Softmax(f1.w2+b2)

--------------------------------------------------------------------------
We have defined Layer dense class : creates instance of layer
this takes 2 arugements __init__(inputs,layers)
and then initiliase a random weight matrix and zero bias matrix

2nd method is forward , gives output of the layer
it will take dot product of bias and weights and add bias term to it
---------------------------------------------------------------------------
Now the output of this class will be fed to ReLU
We have the ReLU activation class defined
this has function called forward which takes max of 0 and input
--------------------------------------------------------------------------
Now the output of this will be fed into w2 matrix
create 2nd layer dense layer
where you again __init__(intialise weight and bias matrix)
then we take dot product of earlier layer f1 and w2+b2
this will be done by 2nd layer dense class
---------------------------------------------------------------------------
Now the output of this will be fed as input to Softmax class
we have already defined that 
it first takes exponent and then divides by the summation
this is the last class we implement

"""





 

