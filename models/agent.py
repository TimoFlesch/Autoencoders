"""
implements agent class, currently not in use
Timo Flesch, 2017
"""

import os
import numpy as np 
import tensorflow as tf 

FLAGS = tf.app.flags.FLAGS

class myAgent(object):
	""" implements agent """
	def __init__(self, 
		        environment = 'env_name', # environment for agent
		        nnet        = None,       # the function approximator
		        epsilon     = None):       # the learning rate
		self.env  = environment
		self.nnet = nnet
		self.lr   = epsilon

		self.results = {
		    'avg_score':  [],
		    'avg_loss':   [],
		    'avg_frames': []
		}

	def train(self,n_runs=1,n_save=1):
		""" trains agent on several episodes and updates nnet 
			- n_runs: (int) number of training runs
			- n_save: (int) number of runs until save ckpt
		"""
		# define result dict
		results = {
		   'score'  = [],
		   'frames' = [],
		   'loss'   = []
		}
		
		# set session
		sess = self.nnet.session
		
		# begin with training
		for ii in range(n_runs):
			score  = 0
			frames = 0 
			# perform experiment 
			done = 0
			while not(done):
				# do sth
				if done:
					results['score'].append(score)
					results['frames'].append(frames)
					results['loss'].append(loss)
				else:
					pass	
			if ii%n_save:
				self.nnet.save_ckpt()
				pass
		return results


	def eval(self,n_runs=1, is_rnd=None):
		""" evaluates agent on several episodes 
			- n_runs: (int)  number of independent runs
			- is_rnd: (bool) random play or not
		"""

		# define result dict
		results = {
		   'score'  = [],
		   'frames' = [],
		   'loss'   = []
		}

		# set session
		sess = self.nnet.session
		
		# begin with evaluation
		for ii in range(n_runs):
			score  = 0
			frames = 0 
			# perform experiment 
			done = 0
			while not(done):
				# do sth
				if done:
					results['score'].append(score)
					results['frames'].append(frames)
					results['loss'].append(loss)
				else:
					pass	

		return results
