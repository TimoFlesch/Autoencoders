import numpy      as np 
import tensorflow as tf
from datetime import datetime

FLAGS = tf.app.flags.FLAGS


def trainModel(sess,nnet,x_train,x_test,n_episodes=100,n_batches=10,batch_size=128,model_dir='./'):
	sess = nnet.session;

	# declare some log vars	
	results = {}
	images_orig          = []
	images_reconstructed = []
	loss_total_train     = []
	loss_total_test      = []
	
	# for memory reasons, split test set into minibatches
	minibatches_xTest = [x_test[k:k+batch_size] for k in range(0,x_test.shape[0],batch_size)]
	# create set of example inputs
	x_examples = x_test[:10]
	# baseline stats (untrained model):
	# 1. add a few test inputs to container
	images_orig.append(x_examples)

	# 2. add a few test reconstructions to container
	images_reconstructed.append(nnet.inference(x_examples))

	# 3. add baseline summaries 	
	nnet.reconstruct(x_test[0,:]) # reconstructed image of only noise

	# --- main training loop ---
	for ep in range(n_episodes):
		# save model
		nnet.save_ckpt(model_dir + '/')	
		
		# shuffle training data
		x_train = shuffleData(x_train)

		# iterate through several batches and train! 
		loss_train = 0
		for ii in range(n_batches):
			# sample training data
			x = getRandomSubSet(x_train,batch_size)
			# perform training step
			loss = nnet.train(x,x)		
			loss_train += loss
		loss_train /= n_batches
		loss_total_train.append(loss_train)

		# interim evaluation
		if ((ep%FLAGS.display_step)==0):
			# save model
			nnet.save_ckpt(model_dir + '/')
			# write reconstructed image to summary	
			nnet.reconstruct(x_test[0,:])
			# write a couple of reconstructed images to result container 
			images_reconstructed.append(nnet.inference(x_examples))			
			# iterate through batches and average test loss
			loss_test = 0
			for mb_test in minibatches_xTest:
				loss = nnet.eval(mb_test,mb_test)
				loss_test += loss
			loss_test /= n_batches
			loss_total_test.append(loss_test)

			# log results to stdout
			epTime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			print('{} Episode {} Training Loss {} Test Loss {} \n'.format(epTime, int(ep), loss_train, loss_test))

	# write results to result dictionary and return
	results = { 'loss_train' : loss_total_train,
				'loss_test'  : loss_total_test,
				'img_in'     : images_orig,
				'img_out'    : images_reconstructed	}
	return results  

def getRandomSubSet(x,batch_size=128):
	""" small helper function to retrieve a sample batch """

	sampleStartIDX = np.random.randint(0,len(x)-batch_size)
	return x[sampleStartIDX:(sampleStartIDX+batch_size),:]
	

def shuffleData(x):
    """ helper function, shuffles data """
    ii_shuff = np.random.permutation(x.shape[0])
    # shuffle data    
    x = x[ii_shuff,:]
    return x