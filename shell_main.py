'''
@Author: your name
@Date: 2020-01-20 14:10:46
@LastEditTime : 2020-01-21 14:22:49
@LastEditors  : Please set LastEditors
@Description: In User Settings Edit
@FilePath: /CIKM2018_QMWFLM/shell_main.py
'''
import sys
import os
import subprocess
import time
import numpy as np

batch_size = [80,100,120,140]

learning_rate = [0.001,0.0001,0.00001,0.00000001]

l2_reg_lambda = [0.00001,0.000001,0.000001]

margin_lambda = [0.01,0.05,0.1]

num_filters = np.arange(20,200,5)

embedding_dim = [50,100,200,300]
dataset = 'wiki'
count = 0

for num_f in num_filters:
	count += 1
	print( 'The count:{} excue'.format(count))
	if dataset == 'trec':
		subprocess.call('python train.py --data trec --clean False --num_filters %d' % num_f,shell = True)
	else:
		subprocess.call('python train.py --data wiki --clean True --num_filters %d' % num_f,shell = True)
		
# for dim in embedding_dim:
# 	count += 1
# 	print( 'The count:{} excue'.format(count))
# 	if dataset == 'trec':
# 		subprocess.call('python train.py --data trec --clean False --embedding_dim %d' % dim,shell = True)
# 	else:
# 		subprocess.call('python train.py --data wiki --clean True --embedding_dim %d' % dim,shell = True)
# for batch in batch_size:
# 	for rate in learning_rate:
# 		for l2 in l2_reg_lambda:
# 			for margin in margin_lambda:

# 				print 'The ', count, 'excue\n'
# 				count += 1
# 				if dataset == 'trec':
# 					subprocess.call('python train.py --data trec --clean False --batch_size %d --learning_rate %f --l2_reg_lambda %f --margin %f' % (batch,rate,l2,margin), shell = True)
# 				else:
# 					subprocess.call('python train.py --data wiki --clean True --batch_size %d --learning_rate %f --l2_reg_lambda %f --margin %f' % (batch,rate,l2,margin), shell = True)