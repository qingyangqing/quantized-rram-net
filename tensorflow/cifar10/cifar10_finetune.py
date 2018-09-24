# Qing 

import tensorflow as tf
import numpy as np
import cifar10
import os
import pdb
from quantizing_ops import initialize_uninitialized, get_weights_list, weights_statistics, \
	generate_weights_b, initialize_weights_c, backup_weights, quantize, quantize_atomic

"""
using default BATCH_SIZE = 128
"""
os.environ["CUDA_VISIBLE_DEVICES"]='0'
cifar10.maybe_download_and_extract()
# Get images and labels for CIFAR-10.
# Force input pipeline to CPU:0 to avoid operations sometimes ending up on
# GPU and resulting in a slow down.
with tf.device('/cpu:0'):
	images_train, labels_train = cifar10.distorted_inputs()

#quant_list = [16, 22, 19, 13, 13]
#quant_list = [0.1*it for it in quant_list]

logits = cifar10.inference(images_train)
accuracy = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels_train, 1), tf.float32))
loss = cifar10.loss(logits, labels_train)

saver = tf.train.Saver(max_to_keep=100)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
	# coordinate threads for all queue runners 
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	# load the pretrained model 
	#saver.restore(sess, './models/my_model_300000')
	saver.restore(sess, './quant_models/my_model_210000')
	# weights statistics
	weights = get_weights_list()
	#means, std_deviations = weights_statistics(sess, weights)
	trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	trainable_variables = list(set(trainable_variables)-set(weights))
	#pdb.set_trace()
	#quant_list = [std*quant for std, quant in zip(std_deviations, quant_list)]
	#weights_b = generate_weights_b(sess, weights)
	#weights_c = initialize_weights_c(sess, weights, quant_list)
	#backup_weights_op = backup_weights(weights, weights_b)
	#quantize_op = quantize(weights, weights_b, weights_c, quant_list)

	# training op config
	max_itr = int(300e3)
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(0.01, global_step=global_step, 
					decay_steps=int(100e3), decay_rate=0.1, staircase=True)
	#learning_rate = tf.constant(0.001)
	train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=trainable_variables, global_step=global_step)
	#train_op = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss, global_step=global_step)
	initialize_uninitialized(sess)

	#sess.run(quantize_op) # let weights_c==weights at the very beginning.
	#pdb.set_trace()
	for itr in range(max_itr):
		#sess.run(backup_weights_op)
		train_op.run()
		#sess.run(quantize_op)
		# save model
		if (itr+1)%10e3 == 0:
			print("save a checkpoint at itration %d" % (itr+1))
			saver.save(sess, './quant_tuneB_models/my_model_'+str(int(itr+1)))
		# print info
		if (itr+1)%100 == 0:
			lr, batch_acc, batch_loss = sess.run([learning_rate, accuracy, loss])
			print("iteration %g, learning rate %g, batch accuracy %g, batch loss %g" % 
				((itr+1), lr, batch_acc, batch_loss))
	# stop all queue runners
	coord.request_stop()
	coord.join(threads)

