# Qing

import tensorflow as tf
import numpy as np
import cifar10
import os
import pdb
from quantizing_ops import get_weights_list

os.environ["CUDA_VISIBLE_DEVICES"]='0'
cifar10.maybe_download_and_extract()
BATCH_SIZE = 100
cifar10.FLAGS.batch_size = BATCH_SIZE
with tf.device('/cpu:0'):
	images_test, labels_test = cifar10.inputs(eval_data=True)

logits = cifar10.inference(images_test)
accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels_test, 1), tf.float32))

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
	# coordinate threads for all queue runners 
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	
	max_itr = int(np.ceil(10000/BATCH_SIZE))
	accl = []
	for idx in range(1, 31):
		# load variables
		#saver.restore(sess, './models/my_model_300000')
		#saver.restore(sess, './quant_models/my_model_'+str(int(10e3*idx)))
		saver.restore(sess, './quant_tuneB_models/my_model_'+str(int(10e3*idx)))
		#weights = get_weights_list()
		#pdb.set_trace()
		# check accuracy
		acc_sum = 0
		for _ in range(max_itr):
			acc_sum += sess.run(accuracy)
		accl.append(acc_sum/max_itr)
		print("test accuracy %g" % accl[-1])
	print("max accuracy %g, model %d" % (np.max(accl), np.argmax(accl)+1))
	# stop all queue runners
	coord.request_stop()
	coord.join(threads)

