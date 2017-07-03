import numpy      as np 
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS


def trainModel(sess,nnet,x_train,x_test,n_episodes=100,n_batches=10,batch_size=128,model_dir='./'):
	sess = nnet.session;

	for ep in range(n_episodes):
		# shuffle training data

		# iterate through several batches 
		for ii in range(n_batches):
			# sample training data
			x = getRandomSubSet(x_train,batch_size)
			# perform training step
			loss = nnet.train(x,x)
			print('Episode {} Batch {} Step {} Training Loss {} \n'.format(ep,ii,nnet.ITER,loss))
		if (ep%FLAGS.display_step):
			# save model
			nnet.save_ckpt(model_dir + '/')
			print('Episode {} Training Loss {} \n'.format(int(ep),loss))
			# evaluate model			
			# 1. make batches (for memory reasons)
			# 2. iterate through batches and average loss


def getRandomSubSet(x,batch_size=128):
	""" small helper function to retrieve a sample batch """

	sampleStartIDX = np.random.randint(0,len(x)-batch_size)
	return x[sampleStartIDX:(sampleStartIDX+batch_size),:]
	