"""
implements model class
Timo Flesch, 2017
"""

import os
import numpy as np 
import tensorflow as tf 

from tfio.io import saveMyModel

FLAGS = tf.app.flags.FLAGS

class myModel(object):
	"""
	the model object in which all its logic is specified
	"""

	def __init__(self,is_trained=False):
		""" initializes the model with parameters """
		if not is_trained:
			with tf.name_scope('input'):
				self.x = tf.placeholder(tf.float,[None,1,784],name='x_in')
				self.y = tf.placeholder(tf.float,[None,1,10],name='label')
				self.y_ = tf.placeholder(tf.float,[None,1,10],name='prediction')

			with tf.variable_scope('nnet'):
				self.create_nnet()
		else:
			self.init_done = True 

	def create_nnet():
		""" creates neural network function approximator """
		# insert magic


	def inference(self,x):
		""" forward pass of x through myModel """
		return modelOP(x,name="inference")


	def loss(self,x_in,y=None):
		""" loss function, e.g. between predicted and true label """
		self.y_ = self.inference(x_in)

			
		with tf.name_scope('loss_funct'):
			# supervised:
			#self.loss = tf.loss(y,y_hat,name="loss")
			# unsupervised (e.g. autoencoder)
			self.loss = tf.loss(x_in,y_hat,name="loss")
			

		
	def optimize(self, batch_x, batch_y):
		""" the optimisation procedure, e.g. SGD """
		with tf.name_scope('Optimizer'):	
			
    	return tf.train.optimizer.minimize(self.loss, name="optimizer")


	def init_graph_vars(self,sess,log_dir='.'):
		""" initializes graph variables """
		# set session
		self.session = sess
		# initialize all variables
		self.init_op = tf.global_variables_initializer()
		self.session.run(self.init_op)

		# define saver object AFTER variable initialisation
		self.saver = tf.train.Saver()

		# define summaries
		tf.summary.scalar("loss",self.loss)
		tf.summary.scalar("learning_rate",self.lr_op)
		self.merged_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(log_dir,self.session.graph)

		self.init_done = true


	def save_ckpt(self,):
		""" saves model checkpoint """
		# save the whole statistical model
		saved_ops = [(self.y_hat,'nnet')] 

        saveMyModel(self.session,
                    self.saver,
                    saved_ops,
                    log_dir='.')
