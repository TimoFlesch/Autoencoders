def trainModel(sess,nnet,x_train,x_test,n_steps=1000,batch_size=128,model_dir='./'):
	sess = nnet.session;

	for ep in range(n_steps):
		# sample training data
		# perform training step

		# every nth step, evaluate on test data
		x = 1;
		if (ep%100):
			nnet.save_ckpt(model_dir)

	