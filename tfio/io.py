import tensorflow as tf


def loadMyModel(sess, op_ids, ckpt_dir):
	ckpt = tf.train.get_checkpoint_state(ckpt_dir)
	if ckpt and ckpt.model_checkpoint_path:
		saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
		saver.restore(sess,ckpt.model_checkpoint_path)

	saved_ops = []
	for ii,op in enumerate(op_ids):
		saved_ops.append(tf.get_collection(op)[0])
	return saved_ops 

def saveMyModel(sess,saver,ops,globalStep=1,model_dir='./'):
	""" saves selected model ops.
		ops are tuples ('name',op)
    """
	for op in ops:
		tf.add_to_collection(op[0],op[1])
	
	saver.save(sess,model_dir,global_step=globalStep)

	return True