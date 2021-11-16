#Introductory workbook going over the fundamentals of TensorFlow
#
# copy of code stored in Google collab
#
#1.   Manipulation of tensors
#2.   Tensors and numpy
#3. using the @tf.function (faster python functions)
#4. utilising GPU/TPU



import tensorflow as tf
import numpy as np

#Creating tensors with tf.constant()
scalar = tf.constant(2)

#Checking dimensionality
scalar.ndim

#create vectors
vec2 = tf.constant([1,2])
vec2.ndim

#create matrix
mat = tf.constant([[1,2],
                   [1,2]])
mat.ndim

another_mat =tf.constant([[1.,2.],
                          [1.,2.]] ,dtype =tf.float16)

#create tensors via variable
changeable_tensor = tf.Variable([10,7])
nonchangable_tensor = tf.constant([10,7])
#use tf.assign to modify variable tensor

#random tensor
rand_tensor = tf.random.Generator.from_seed(42) #set seed for reproducibility this uses a uniform probability
rand_tensor = rand_tensor.uniform(shape=(3,2))

rand_tensor2= tf.random.Generator.from_seed(76)
rand_tensor2= rand_tensor2.normal(shape=(3,2)) # this uses a normal distibution of probabilities


#shuffling a tensor along its first dimension
not_shuffled = tf.constant([[10,7],
                            [9,2],
                            [3,4]])

tf.random.shuffle(not_shuffled)

#numpy arrays can be converted to tensorflow tensors which are more efficiently run on a gpu

numpy_A  = np.arange(1,25, dtype=np.int32)
A = tf.constant(numpy_A)
A