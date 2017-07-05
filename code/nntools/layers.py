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
"""
wrapper for convolution
"""
	
	with tf.variable_scope(name):
		
		kernel = [filter_size[0],filter_size[1],x.get_shape()[-1],n_filters]
		stride = [1,stride[0],stride[1],1]
		shape = x.get_shape().as_list()

		weights = tf.get_variable('weights',kernel,tf.float32,initializer=initializer)
		biases = tf.get_variable('biases',[n_filters],initializer=tf.constant_initializer(bias_const))
		convolution = tf.nn.conv2d(x,w,stride,padding,data_format='NHWC')

		y  = tf.nn.bias_add(convolution,biases,'NHWC')
		
		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases, shape




def layer_transpose_conv2d(x,
				n_filters   =       32,
				filter_size =    (6,6),
				stride      =    (1,1),
				shape       = [1,2,2,2],
				padding     =  'VALID',
				name        = 'transconv2d',
				bias_const  =      0.0,
				initializer = tf.contrib.layers.xavier_initializer(),
				nonlinearity =   None):
"""
wrapper for transposed convolution
"""
	
	with tf.variable_scope(name):
		kernel = [filter_size[0],filter_size[1],x.get_shape()[-1],n_filters]
		stride = [1,stride[0],stride[1],1]
		out_dims = tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]])
		weights = tf.get_variable('weights',kernel,tf.float32,initializer=initializer)
		biases = tf.get_variable('biases',[n_filters],initializer=tf.constant_initializer(bias_const))
		convolution = tf.nn.conv2d_transpose(x,w,stride,padding,data_format='NHWC')

		y  = tf.nn.bias_add(convolution,biases,'NHWC')
		
		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases 