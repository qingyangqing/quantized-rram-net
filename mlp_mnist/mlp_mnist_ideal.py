# Qing 
# 22nd Mar 2017

import caffe
import numpy as np
import copy
from caffe import layers as L, params as P
import pdb
import os 

caffe.set_device(0)
caffe.set_mode_gpu()

# define net structor function
def mknet(lmdb, batch_size, gg_max):
    n = caffe.NetSpec()
    n.data, n.label = L.Data(batch_size=batch_size, backend=P.Data.LMDB, source=lmdb,
				transform_param=dict(scale=1./255), ntop=2)
    n.fc1 =   L.InnerProduct(n.data, num_output=128, 
				weight_filler=dict(type='gaussian', std=gg_max),
				bias_filler=dict(type='constant'))
    n.relu1 = L.ReLU(n.fc1, in_place=True)
    n.fc2 = L.InnerProduct(n.relu1, num_output=10, 
				weight_filler=dict(type='gaussian', std=gg_max),
				bias_filler=dict(type='constant'))
    n.loss =  L.SoftmaxWithLoss(n.fc2, n.label)
    n.acc = L.Accuracy(n.fc2, n.label)
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
with open('mlp_train.prototxt', 'w') as f:
    f.write(str(mknet(cwd+'/../data/mnist/mnist_train_lmdb', 60, gg_max)))
with open('mlp_test.prototxt', 'w') as f:
    f.write(str(mknet(cwd+'/../data/mnist/mnist_test_lmdb', 100, gg_max)))
# initialize solver 
solver = caffe.get_solver(cwd+'/mlp_solver.prototxt')
# training 
nitr=1000*100
test_interval=1000
test_acc = np.zeros(nitr/test_interval)
for i in range(nitr):
	# one step training
	solver.step(1)
	# run a full test every so often
	if (i+1) % test_interval == 0:
		print 'Iteration', i, 'testing...' 
		correct = 0
		for _ in range(100):
			solver.test_nets[0].forward()
			correct+=solver.test_nets[0].blobs['acc'].data
		print 'Test accuracy: ', correct/100
		test_acc[((i+1)/test_interval)-1] = correct/100	

np.save('acc_ideal.npy', test_acc)
# solver.net.save('ideal.caffemodel')
temp_params=copy.deepcopy(
	[[k, v[0].data[...], v[1].data[...]] for k,v in solver.net.params.items()])
# quantization
params_q=quantization(temp_params, gg)
set_params(solver.net, params_q)
correct = 0
for _ in range(100):
	solver.test_nets[0].forward()
	correct+=solver.test_nets[0].blobs['acc'].data
print 'Test accuracy after direct quantization: ', correct/100
