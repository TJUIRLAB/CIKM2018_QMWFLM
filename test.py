'''
@Author: your name
@Date: 2020-01-20 14:10:46
@LastEditTime: 2020-01-20 15:28:06
@LastEditors: your name
@Description: In User Settings Edit
@FilePath: /CIKM2018_QMWFLM/test.py
'''
import tensorflow as tf 
import cPickle as pickle
import numpy as np 
import pandas as pd

from scipy.stats import ttest_ind, levene, wilcoxon
# a = tf.Variable(np.ones((3,33,10)))
# b = tf.expand_dims(tf.Variable(np.arange(33) + 0.0),-1)
# print b
# c = tf.transpose(a,perm = [1,0]) * b
# c = tf.multiply(a,b)
# d = tf.ones([10,2])
# a = np.arange(3 * 3 * 10).reshape(3,3,10)
# b = np.arange(3 * 3 * 1).reshape(3,3,1)
# c = tf.multiply(a,b)
# # c = tf.reduce_sum(b**2)
# # initializer = (np.array(0), np.array(1))
# # fibonaccis = tf.scan(lambda a, _: (a[1], a[0] + a[1]), elems)
# with tf.Session() as sess:

# 	sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
# 	print sessnp.arange(100).run(c)
# array = np.arange(3 * 9).reshape(3,3,3)
# print array
# h = tf.trace(array)
# with tf.Session() as sess:
# 	print sess.run(h)
# np.random.shuffle(array)
# print(np.log(10e5))
# print sorted(array)
	# print sess.run(d)
# file1 = 'ap_score_qlm'
# file2 = 'ap_score_wiki'


# ap_1 = pd.read_csv('ap_score',header = None,names = ['question','ap'],sep = '\t')
# ap_2 = pd.read_csv('ap_score_wiki', header = None,names = ['question','ap'],sep = '\t')

# print wilcoxon(ap_1['ap'], ap_2['ap'])
# import numpy as np
# import matplotlib.pyplot as plt
# # alpha = ['ABC', 'DEF', 'GHI', 'JKL']
# d = pickle.load(open('attention.file'))
# print d[0][0]
# exit()
# # print len(d)
# data = d[0][0]
# print data
# # print d[0][0]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# cax = ax.matshow(data, cmap = plt.cm.Blues)
# fig.colorbar(cax)

# ax.set_xticklabels(['']+alpha)
# ax.set_yticklabels(['']+alpha)

# plt.show()

# a = []

# b = np.ones((10,10))
# c = np.random.rand(10,20)
# print c[0]
# for b1,c1 in zip(b,c):
# 	a.extend((b1,c1))

# print a[1]
# import pandas as pd 
# file = 'data/nlpcc/train.txt'
# df = pd.read_csv(file,header = None,sep="\t",names=["question","answer","flag"],quoting =3).fillna('')
# df['alen'] = df.apply(lambda x:len(x['answer'].split()),axis = 1)
# print df[df['flag'] == 1]['alen'].
# a = ('a','b')
# print str(a)

a = [1,2,3,4,5]
b = [5,6,7,8,9]

print  (np.asarray(a).dot(np.asarray(b)))