
"""
Tensors in PyTorch
Tensors are a data structure: to hold, store, operate- 
specialised multi dimensional array for computational efficiency 
Tensor performs a task in n-dimensions which an array does in 1 D

Dimension : it suggests in how many directions a tensor spans

for example, vector is spanned (spread) in 1D
Matrix is 2D spanned
Cube is 3D spanned

Every tensor has an associated dimension with it

Scalars : Scalar means a single number (for ex, 2)
they are 0D tensors or scalars
for ex : in forward pass NN, the loss function computes a single value indicating 
the difference between predicted and actual outputs : 0 D tensor

Vectors : 1D tensors
represents seq of value
For example, in embedding a token gets tokenized into an embedding "Vector"
this is an example of 1D vector

Matrix: Image : 2D grid of numbers : grayscale image
why not RGB image, becuase it has 3 channels structure


3D tensors : RGB image (coloured image)
it has a 3 channel structure  : RGB
for example an RGB image (eg : 256 x 256) : Shape represented : [256,256,3]


4D tensors : batch of RGB images
when neural nets are trained, you dont send a single value or image
you send a batch of values or images (for example, 32 at a shot)
Example : A batch of 32 images each of 128 X 128 with 3 colour channels (RGB)
would have shape (batch size x width x height x channels)
In this case : [32,128,128,3]


5D tensors : Video data
Video has frames , and every frame is an image
imagine a stream of snapshots or combination of images

Imagine you batch pass a string of 10 video clips to a neural net
each with 16 frames of size 64 x 64 and 3 channels RGB would have shape
[10,16,64,64,3]
#[10 videos, 16 frames per video, every frame is RGB 64x64 , number of channels]

You can represent higher dimensions in tensor (n-D)
But vectors are limited to 1-D

Now think of multimodal LLMs and their input-output in tensors


** Why are Tensors Useful ? **

We can easily perform common math ops in neural nets using tensors
super efficient with dot products in neural nets

You can map real world data using tensors in a neural net 
For example, image, audio, video or text
For example, image, audio, video are all matrix of numbers 
i can represent that using tensors

Optimized for efficiency , you can run them on GPUs
Works best with working with multimodal data
Especially for parallel computations on GPUs


** Where are tensors used in DL ? **

Data storage : training data (image,audio,video) is stored in tensors
Weights and Biases : Learanable params of NN  are stored as tensors
WX+B like operations : tensor operations

"""

#%%
# pytorch is preinstalled on google collab
# else: pip install torch

"importing stuff"

import torch
print(torch.__version__)

# checking if GPU available 
# get_device_name  : shows which GPU is available
# as of now shows CPU : If needed GPU : change runtime type : GPU

if torch.cuda.is_available():
    print("GPU is available!")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available. Using CPU.")
    
# as of now GPU not needed, CPU can suffice, but if needed
# copy paste the code in collab, change runtime type to GPU and check

#%%

"Creating a tensor using empty and specify shape"

torch.empty(2,3) #creates a 2x3 matrix

"check type"

a = torch.empty(2,3)
type(a) # it isa tensor object

#%%

"Tensor DataTypes"


"using zeroes and specify shape just like np.zeroes"
torch.zeros(3,3)

torch.ones(3,3)

"torch.rand : make a tensor, all values lie randomly in 0 to 1"
torch.rand(4,3)

"What if we want to maintain reproducibility ? you dont want random values on every execution"
"each time, same output : define seed"

torch.manual_seed(100)
torch.rand(2,3)

"now randomness is curbed"


"Create custom tensors using torch.tensors"

torch.tensor([[1,2,3], [4,5,6]])

#%%
"Some overlapping functions with numpy"

# arange : step 
print("using arange ->", torch.arange(0,10,2))

# using linspace : linearly spaced in a defined range
print("using linspace ->", torch.linspace(0,10,10))

# using eye : identity matrix  : diagonal items are 1 here
print("using eye ->", torch.eye(5))

# using full : define the shape and every item is the same number
print("using full ->", torch.full((3, 3), 5))

#%%

x = torch.tensor([[1,2,3],[4,5,6]])
x

x.shape # 2 rows 3 cols

"What if you want to make a same tensor with same shape ? "

torch.empty_like(x)
# empty_like : same shaepe, different values

torch.zeros_like(x)
# zeros_like : same shape, all items zero

torch.ones_like(x)
# ones_like : same shape, all items one


#%%

"Tensor shapes"

x.dtype


"I want to make a tensor of int dtype"

torch.tensor([1.0, 2.0, 3.0], dtype=torch.int64)

torch.tensor([1,2,3], dtype = torch.float32)

"change data type of a tensor : for example : int to float"

x.dtype

# x is int dtype, we can change it  to float
x.to(torch.float32)

torch.rand_like(x, dtype=torch.float32)

#%%

"Mathematical operations on tensors"


"Scalar operations : between scalar and tensor"



x = torch.rand(2,2)
x

x + 2 # to every tensor item we are adding a scalar (here 2)

x * 3

x - 2

x / 3

#int division
(x*100)//3

#power
x**2

"Element wise operation"

# if you have more than 1 tensors 

a = torch.rand(2,3)
a
b = torch.rand(2,3)
b
# perform item wise operation on tensors with same shape
a+b

a - b

a * b

a/b

# you can explore these basic operations later

"some more element wise operations on a single tensor"

c = torch.tensor([1,-2,-3,4])
c
torch.abs(c)

torch.neg(c) #positive qty becomes negative and negative becomes pos

#round
d = torch.tensor([1.9,2.8,7.2,8.9])
d
torch.round(d) # all are float

# ceil goes to the top value for example 1.9 becomes 2
# floor goes to the bottom value for example 1.9 becomes 1

# ceil
torch.ceil(d)

torch.floor(d)

"Reduction operations"

e = torch.randint(size=(2,3), low=0, high=10, dtype= torch.float32)
e

#sum of all tensors
torch.sum(e)
# sum along the columns
torch.sum(e, dim=0)

#sum along the rows
torch.sum(e,dim=1)

# mean
torch.mean(e) 

# mean along the columns
torch.mean(e, dim=0) # 0 is for column basis

torch.median(e, dim = 0)
# indices=tensor([0, 1, 0])) , index position of the median value found in that matrix

#product
torch.prod(e)

#standard deviation
torch.std(e)

#variance
torch.var(e)

# argmax : position of the largest item in the matrix
e

torch.argmax(e)

torch.argmin(e)


"Matrix Operations"

# take 2 matrices : 2x3 and 3x2

g = torch.randint(size=(2,3), low=0, high=10)
h = torch.randint(size=(3,2), low=0, high=10)

print(g)
print(h)

# matrix multiplication
torch.matmul(g,h)

#dot product between 2 vectors
vector1 = torch.tensor([1, 2])
vector2 = torch.tensor([3, 4])

# dot product
torch.dot(vector1, vector2)

"Comparison operations"

i = torch.randint(size=(2,3), low=0, high=10)
j = torch.randint(size=(2,3), low=0, high=10)

print(i)
print(j)


# greater than
i > j

# less than
i < j

# equal to
i == j

# not equal to
i != j

# greater than equal to

i>=j

# less than equal to
i <= j 


"Special functions - log, exp , etc..."

k = torch.randint(size=(2,3), low=0, high=10, dtype=torch.float32)
k

# log
torch.log(k)


# exp
torch.exp(k)


# sqrt
torch.sqrt(k)


# sigmoid
torch.sigmoid(k)

"Copying a Tensor"
a = torch.rand(2,3)
a

b = a # assignment operator

b

# but there is a problem with assignment operator
# any changes that you make in the original are also reflected in the 
# copy , and sometimes this is not desirable
# for example
a[0][0] = 0
a

#now check b , they both are same
b

# check id function
id(a) # where in memory is the tensor stored
id(b)
# they both are mapped to the same id

# in that case, prefer using clone function

b = a.clone()

print(a)

print(b)

# now change the value in a 
a[0][0] = 10
print(a)

# now check b , it will be different
print(b)

# checking the id memory locations

id(a)
id(b)

# we can see they are both mapped to different id 


# you can now checkout the notebook on Tensor operations on GPU

# link :https://colab.research.google.com/drive/1rBPyS6KuZYraE19rITFaU0OuGtymOwns#scrollTo=OmhhJDyDw70E




