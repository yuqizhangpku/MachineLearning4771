# get tf as the feature

import random
from math import log10
from classfile import TFIDFCOUNTER, OnlinePerceptron

train_example = 1000000

def count_accuracy(train_example):
	# tfcount = TFCOUNTER()
	tfcount = TFIDFCOUNTER()
	# tfcount = BIGRAMCOUNTER()
	onlinepreceptron = OnlinePerceptron()
	f = open('../reviews_tr.csv', 'r')
	lines = f.readlines()
	f.close()
	word_set = {}
	data = []
	lable = []
	cnt = 0
	total_cnt = len(lines)

	# the first pass 
	for line in lines[1:train_example]:
		line = line.strip().split(',')
		lable.append(line[0])
		text = line[1]
		data.append(tfcount.get_tf(text))
		# print('tf\t%s'%(tfcount.get_tf(text)))
		if cnt%100 == 0:
			onlinepreceptron.train(data, lable, 0.1)
			data = []
			lable = []
			print(float(cnt)/float(total_cnt))
		cnt += 1
	onlinepreceptron.train(data, lable, 0.1)

	# the second pass 
	data = []
	lable = []
	visit = [0 for _ in range(total_cnt)]
	cnt = 0
	onlinepreceptron.init_turn()
	while cnt < train_example:
		ran = random.randint(1,train_example-1)
		if visit[ran] == 0:
			line = lines[ran].strip().split(',')
			lable.append(line[0])
			text = line[1]
			data.append(tfcount.get_tf(text))
			pre = onlinepreceptron.predict(data[0])
			if pre != int(lable[0]):
				onlinepreceptron.update_final(data[0], cnt)
			onlinepreceptron.train(data, lable, 0.1)	
			data = []
			lable = []
			if cnt%10000 == 0:
				print(float(cnt)/float(total_cnt))
			cnt += 1

	# predict test and count the accuracy
	f = open('../reviews_te.csv', 'r')
	lines = f.readlines()
	f.close()
	data = []
	lable = []
	cnt = 0
	error = 0
	total_cnt = len(lines)
	for line in lines[1:]:
		line = line.strip().split(',')
		lable.append(line[0])
		text = line[1]
		data.append(tfcount.get_tf(text))
		pre = onlinepreceptron.predict(data[0])
		if pre != int(lable[0]):
			error += 1
		onlinepreceptron.train(data, lable, 0.1)	
		data = []
		lable = []
		if cnt%10000 == 0:
			print(float(cnt)/float(total_cnt))
		cnt += 1
	print error
	print cnt
	print float(error)/float(cnt)
	print onlinepreceptron.getbiggest()
	print onlinepreceptron.getsmallest()
	return float(error)/float(cnt)

if __name__ == "__main__":
	count_accuracy(train_example)