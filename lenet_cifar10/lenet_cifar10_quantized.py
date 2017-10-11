# Qing 
# 22nd Mar 2017

import caffe
import numpy as np
import copy
from caffe import layers as L, params as P
import os 

caffe.set_device(0)
caffe.set_mode_gpu()

# define net structor function
def mknet(lmdb, mean_file, batch_size, gg_max):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
				transform_param=dict(mean_file=mean_file, scale=1./128), ntop=2)
    n.conv1 = L.Convolution(n.data, kernel_size=5, num_output=32, pad=2, stride=1, 
				weight_filler=dict(type='gaussian', std=gg_max),
				bias_filler=dict(type='constant'))
    n.relu1 = L.ReLU(n.conv1, in_place=True)
    n.pool1 = L.Pooling(n.relu1, kernel_size=3, stride=2, pool=P.Pooling.MAX)
    n.norm1 = L.LRN(n.pool1, local_size=3, alpha=5e-5, beta=0.75, norm_region=1)
    n.conv2 = L.Convolution(n.norm1, kernel_size=5, num_output=32, pad=2, stride=1,
				weight_filler=dict(type='gaussian', std=gg_max),
				bias_filler=dict(type='constant'))
    n.relu2 = L.ReLU(n.conv2, in_place=True)
    n.pool2 = L.Pooling(n.relu2, kernel_size=3, stride=2, pool=P.Pooling.AVE)
    n.norm2 = L.LRN(n.pool2, local_size=3, alpha=5e-5, beta=0.75, norm_region=1)
    n.conv3 = L.Convolution(n.norm2, kernel_size=5, num_output=64, pad=2, stride=1,
				weight_filler=dict(type='gaussian', std=gg_max),
				bias_filler=dict(type='constant'))
    n.relu3 = L.ReLU(n.conv3, in_place=True)
    n.pool3 = L.Pooling(n.relu3, kernel_size=3, stride=2, pool=P.Pooling.AVE)
    n.fc1 =   L.InnerProduct(n.pool3, num_output=10, 
				weight_filler=dict(type='gaussian', std=gg_max),
				bias_filler=dict(type='constant'))
    n.loss =  L.SoftmaxWithLoss(n.fc1, n.label)
    n.acc = L.Accuracy(n.fc1, n.label)
    return n.to_proto()

# define quantization function
def quantization(params_data, gg):
	params_q=copy.deepcopy(params_data)
	length=len(gg)
	gg_inter=(gg[0:length-1]+gg[1:length])/2
	length-=1
	for i in range(len(params_data)):
		if(params_data[i][0][:4]!='norm'): # if not norm layer 
			for it in [1,2]:
				temp=params_data[i][it]
				data_q=np.zeros(temp.shape)
				data_q+=gg[0]*(temp<gg_inter[0])
				for j in range(1, length):
					data_q+=gg[j]*((temp>=gg_inter[j-1])*(temp<gg_inter[j]))
				data_q+=gg[length]*(temp>=gg_inter[length-1])
				params_q[i][it]=data_q
	return params_q
# define function set_params
def set_params(net, params_data):
	i=0
	for k,_ in net.params.items():
		if (k[:4]!='norm'): 
			net.params[k][0].data[...]=params_data[i][1]
			net.params[k][1].data[...]=params_data[i][2]
		i+=1

# gg choice
res_levels=2
single_g=np.linspace(0.05, 1.0, num=res_levels)/10
length=len(single_g)
gg_mat=np.zeros([length, length])
for i in range(length):
	gg_mat[i,:]=single_g[i]-single_g
gg=np.unique(gg_mat)
gg.sort()
gg_max = max(gg) 
# make lenet 
cwd = os.getcwd()
mean_file = cwd+'/../data/cifar10/mean.binaryproto'
with open('lenet_train.prototxt', 'w') as f:
    f.write(str(mknet(cwd+'/../data/cifar10/cifar10_train_lmdb', mean_file, 100, gg_max)))
with open('lenet_test.prototxt', 'w') as f:
    f.write(str(mknet(cwd+'/../data/cifar10/cifar10_test_lmdb', mean_file, 100, gg_max)))
# initialize solver 
solver = caffe.get_solver('lenet_solver.prototxt')
# initialize net parameters  
params_c = copy.deepcopy(
	[[k, v[0].data[...], v[1].data[...]] for k,v in solver.net.params.items()])
# quantization
params_q = quantization(params_c, gg)
params_c = params_q 
set_params(solver.net, params_q)

# training 
nitr=1000*100
test_interval=1000
test_acc = np.zeros(nitr/test_interval) 
for i in range(nitr):
	# get params value
	params=copy.deepcopy(
		[[k, v[0].data[...], v[1].data[...]] for k,v in solver.net.params.items()])
	# one step training
	solver.step(1)
	temp_params=copy.deepcopy(
		[[k, v[0].data[...], v[1].data[...]] for k,v in solver.net.params.items()])
	# loss of mini-batch 
	loss=solver.net.blobs['loss'].data
	# update continuous params
	for layer_num in range(len(temp_params)):
		if(temp_params[layer_num][0][:4]!='norm'):
			for it in [1,2]:	
				delta=temp_params[layer_num][it]-params[layer_num][it]
				params_c[layer_num][it]+=delta
				# clipping 
				temp=params_c[layer_num][it]
				temp=temp*(abs(temp)<=gg_max)+gg_max*(temp>gg_max)-gg_max*(temp<-gg_max)
				params_c[layer_num][it]=temp
	# quantize updated continuous params
	temp_params=quantization(params_c, gg)
	# update net if needed
	set_params(solver.net, temp_params)
	solver.net.forward(start='conv1')
	if solver.net.blobs['loss'].data>loss:
		set_params(solver.net, params)
	# run a full test every so often
	if (i+1) % test_interval == 0:
		print 'Iteration', i, 'testing...' 
		correct = 0
		for _ in range(100):
			solver.test_nets[0].forward()
			correct += solver.test_nets[0].blobs['acc'].data
		print 'Test accuracy: ', correct/100
		test_acc[((i+1)/test_interval)-1] = correct/100			
# # save caffemodel 
# solver.net.save(str(res_levels)+'.caffemodel')
# save test accuracy 
np.save('acc_quantized.npy', test_acc)
