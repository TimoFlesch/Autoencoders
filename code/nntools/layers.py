"""
wrapper function for frequently-used tensorflow layers
Timo Flesch, 2017
"""
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers 


def layer_fc(x,
			dim_y,			
			bias_const   =      0.0,
			name         = 'linear',
			initializer  = tf.random_normal_initializer(0.05),
			nonlinearity =    None):

	with tf.variable_scope(name):
		weights = tf.get_variable('weights',[x.get_shape().as_list()[1],dim_y],tf.float32, initializer=initializer)

		biases  = tf.get_variable('biases',[dim_y],initializer=tf.constant_initializer(bias_const))

		y = tf.nn.bias_add(tf.matmul(x,weights),biases)
		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases



def layer_conv2d(x,
				n_filters   =       32,
				filter_size =    (6,6),
				stride      =    (1,1),
				padding     =  'VALID',
				name        = 'conv2d',
				bias_const  =      0.0,
				initializer = tf.contrib.layers.xavier_initializer(),
				nonlinearity =   None):
	
	with tf.variable_scope(name):
		kernel = [x.get_shape()[-1],filter_size[0],filter_size[1],n_filters]
		stride = [1,stride[0],stride[1],1]

		weights = tf.get_variable('weights',kernel,tf.float32,initializer=initializer)
		biases = tf.get_variable('biases',[n_filters],initializer=tf.constant_initializer(bias_const))
		convolution = tf.nn.conv2d(x,w,stride,padding,data_format='NHWC')

		y  = tf.nn.bias_add(convolution,biases,'NHWC')
		
		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases 