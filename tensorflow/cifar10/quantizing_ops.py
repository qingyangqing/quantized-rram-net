# Qing

import tensorflow as tf
import numpy as np

"""
get weights variables list
"""
def get_weights_list():
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weights_variables = [n for n in trainable_variables if 'weights' in n.name]
        print([n.name for n in weights_variables])
        return weights_variables

"""
explore means and standard deviations for each layer's weights
"""
def weights_statistics(sess, weights):
        means, std_deviations = [], []
        for it in weights:
                flattened = tf.reshape(it, [-1])
                means.append(sess.run(tf.keras.backend.mean(flattened)))
                std_deviations.append(sess.run(tf.keras.backend.std(flattened)))
        return means, std_deviations

"""
quantizing atomic op
-- W: a weight variable
-- Q: a scalar to be quantized to
"""
def quantize_atomic(W, Q):
	Q = tf.cast(Q, tf.float32)
	Wb = tf.zeros(tf.shape(W))
	Wb = tf.add(Wb, tf.multiply(-Q, tf.cast(tf.less(W, -Q/2), tf.float32)))
	Wb = tf.add(Wb, tf.multiply( Q, tf.cast(tf.less_equal(Q/2, W), tf.float32)))
	return Wb

"""
initialize a list of variables weights_c
-- weights_list
-- quantization value list
"""
def initialize_weights_c(sess, weights, quant_list, name=None):
        with tf.name_scope(name, "initialize_weights_c") as name:
                weights_c = [tf.Variable(quantize_atomic(weights[i], quant_list[i])) for i in range(len(weights))]
                sess.run(tf.variables_initializer(weights_c))
                return weights_c

"""
generate backup weights_b
"""
def generate_weights_b(sess, weights, name=None):
        with tf.name_scope(name, "generate_weights_b") as name:
                weights_b = [tf.Variable(it) for it in weights]
                sess.run(tf.variables_initializer(weights_b))
                return weights_b

"""
op to backup weights
"""
def backup_weights(weights, weights_b):
        return [tf.assign(weights_b[i], weights[i]) for i in range(len(weights))]

"""
op to quantize weights
"""
def quantize(weights, weights_b, weights_c, quant_list, name=None):
	with tf.name_scope(name, "quantize_weights") as name:
		quantize_ops = []
		for L in range(len(weights)):
			quant_list[L] = tf.cast(quant_list[L], tf.float32)
			delta = weights[L]-weights_b[L]
			weights_c_update = weights_c[L]+delta
			# clipping
			max_mat = tf.multiply(tf.ones(tf.shape(weights_c_update)), quant_list[L])
			min_mat = tf.multiply(tf.ones(tf.shape(weights_c_update)), -quant_list[L])
			weights_c_update = tf.where(tf.less(quant_list[L], weights_c_update), max_mat, weights_c_update)
			weights_c_update = tf.where(tf.less(weights_c_update, -quant_list[L]), min_mat, weights_c_update)
			# quantizing
			W_update = quantize_atomic(weights_c_update, quant_list[L])
			quantize_ops.extend([tf.assign(weights[L], W_update), tf.assign(weights_c[L], weights_c_update)])
		return quantize_ops

def initialize_uninitialized(sess):
        global_vars = tf.global_variables()
        is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        print([str(i.name) for i in not_initialized_vars]) # only for testing
        if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))

