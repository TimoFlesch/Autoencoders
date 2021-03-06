"""
wrapper function for frequently-used tensorflow layers
Timo Flesch, 2017
"""
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers 
from tensorflow.contrib.keras                import layers


def layer_dropout(x,
                keep_prob = 1.0,
                name='dropout'):
    """
    dropout layer
    """
    with tf.variable_scope(name):
        y = tf.nn.dropout(x, keep_prob)
        return y



def layer_flatten(x,
                name = 'flatten'):
    """
    flattens the output of a conv/maxpool layer
    """
    with tf.variable_scope(name):
        # shape_in = x.get_shape().as_list()
        # y = tf.reshape(x,[-1, shape_in[1]*shape_in[2]*shape_in[3]])
        y = tf.contrib.layers.flatten(x)
        return y


def layer_fc(x,
			dim_y,			
			bias_const   =      0.0,
			name         = 'linear',
			initializer  = tf.contrib.layers.xavier_initializer(),#tf.random_normal_initializer(0.05),
			nonlinearity =    None):
	"""
	simple fully connected layer with optional nonlinearity
	"""

	with tf.variable_scope(name):
		weights = tf.get_variable('weights',[x.get_shape().as_list()[1],dim_y],tf.float32, initializer=initializer)

		biases  = tf.get_variable('biases',[dim_y],initializer=tf.constant_initializer(bias_const))

		y = tf.nn.bias_add(tf.matmul(x,weights),biases)
		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases

def layer_fc_bn(x,
			dim_y,			
			bias_const   =         0.0,     # bias 
			name_fc      =    'linear',  # fc var scope
			name_bn      = 'batch_norm', # bn var scope
			phase        =         True, # is_training
			initializer  = tf.contrib.layers.xavier_initializer(),#tf.random_normal_initializer(0.05),
			nonlinearity =    None):
	"""
	simple fully connected layer with Batch Normalisation and optional nonlinearity
	"""

	with tf.variable_scope(name_fc):
		weights = tf.get_variable('weights',[x.get_shape().as_list()[1],dim_y],tf.float32, initializer=initializer)

		biases  = tf.get_variable('biases',[dim_y],initializer=tf.constant_initializer(bias_const))

		y_fc = tf.nn.bias_add(tf.matmul(x,weights),biases)


	with tf.variable_scope(name_bn):
		y = tf.contrib.layers.batch_norm(y_fc,center=True, scale=True, is_training=phase)

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
		convolution = tf.nn.conv2d(x,weights,stride,padding,data_format='NHWC')

		y  = tf.nn.bias_add(convolution,biases,'NHWC')
		
		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases, shape



def layer_max_pool_2x2(x,
                    filter_size = (2,2),
                    stride = (2,2),
                    padding='SAME',
                    name='maxpool_2x2'):

    """
    wrapper for max pool operation
    """
    with tf.variable_scope(name):
        kernel = [1,filter_size[0],filter_size[1],1]
        stride = [1,stride[0],stride[1],1]
        shape = x.get_shape().as_list()

        y = tf.nn.max_pool(x, kernel, stride, padding) 
        return y,shape



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
		kernel = [filter_size[0],filter_size[1],shape[3],n_filters]
		stride = [1,stride[0],stride[1],1]
		out_dims = tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]])
		weights = tf.get_variable('weights',kernel,tf.float32,initializer=initializer)
		biases = tf.get_variable('biases',[shape[3]],initializer=tf.constant_initializer(bias_const))
		convolution = tf.nn.conv2d_transpose(x,weights,out_dims,stride,padding,data_format='NHWC')

		y  = tf.nn.bias_add(convolution,biases,'NHWC')
		shape = x.get_shape().as_list()
		if nonlinearity != None:
			y = nonlinearity(y)

		return y,weights,biases,shape


def layer_upsampling_2D(x,
	filter_size = (2,2),
	name = 'upsampling_2D'):
	"""
	wrapper for Keras' upsampling2D layer
	with import from tn.nn.contrib.keras
	"""

	with tf.variable_scope(name):
		y = layers.UpSampling2D(filter_size)(x)

		return y
