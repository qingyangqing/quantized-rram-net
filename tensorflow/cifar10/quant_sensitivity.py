# Qing
"""
explore each layer's sensitivity for quantization
"""

import tensorflow as tf
import numpy as np
import cifar10
import os
from quantizing_ops import get_weights_list, weights_statistics, \
	generate_weights_b, initialize_weights_c, backup_weights, quantize, quantize_atomic

os.environ["CUDA_VISIBLE_DEVICES"]='2'
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
	saver.restore(sess, 'models/my_model_300000')
	weights = get_weights_list()
	means, std_deviations = weights_statistics(sess, weights)

	quant = tf.placeholder(tf.float32)
	quantize_op = []
	for L in range(len(weights)):
		Wq = quantize_atomic(weights[L], quant)
		quantize_op.extend([tf.assign(weights[L], Wq)])

	max_itr = int(np.ceil(10000/BATCH_SIZE))
	def test_accuracy():
		acc_sum = 0
		for _ in range(max_itr):
			acc_sum += sess.run(accuracy)	
		return acc_sum/max_itr

	print("original accuracy %g" % test_accuracy())
	accl = [[] for _ in range(len(weights))]
	for L in range(len(weights)):
		print("Layer %g" % (L+1))
		for i in range(1, 101):
			saver.restore(sess, 'models/my_model_300000')
			sess.run(quantize_op[L], {quant: std_deviations[L]*0.1*i})
			accl[L].append(test_accuracy())
	#print("final results")
	#[print(it) for it in accl]
	np.save("accl.npy", accl)
	# stop all queue runners
	coord.request_stop()
	coord.join(threads)

