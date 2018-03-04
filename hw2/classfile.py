
from math import log10

# get word's tf
class TFCOUNTER:

	def __init__(self):
		self.feature_word = self.get_feature_word()

	# get the word list
	def get_feature_word(self):
		f = open('word', 'r')
		res = {}
		for line in f.readlines():
			res[line.split()[0]] = 1
		f.close()
		return res

	# count the words in text and get its tf
	def get_tf(self, text):
		words = text.strip().split()
		word_num = len(words)
		word_cnt = {}
		feature = {}
		for word in words:
			tmp = word_cnt.get(word, 0)
			word_cnt[word] = tmp + 1
		for i in word_cnt:
			if i in self.feature_word:
				feature[i] = float(word_cnt[i])
		return feature

# get word's tf-idf
class TFIDFCOUNTER:

	def __init__(self):
		self.total_cnt = 1000000.0
		self.feature_word = self.get_feature_word()

	# get the word and get its idf
	def get_feature_word(self):
		f = open('idfword', 'r')
		res = {}
		for line in f.readlines():
			line = line.strip().split()
			res[line[0]] = self.total_cnt/float(line[1])
		f.close()
		return res

	# get the word's tf-idf
	def get_tf(self, text):
		words = text.strip().split()
		word_num = len(words)
		word_cnt = {}
		feature = {}
		for word in words:
			tmp = word_cnt.get(word, 0)
			word_cnt[word] = tmp + 1
		for i in word_cnt:
			if i in self.feature_word:
				feature[i] = float(word_cnt[i])/word_num*log10(self.feature_word[i])
		return feature

# get bigram words' tf
class BIGRAMCOUNTER:

	def __init__(self):
		self.feature_word = self.get_feature_word()

	# get the bigram word's list
	def get_feature_word(self):
		f = open('biword', 'r')
		res = {}
		for line in f.readlines():
			res[line.split()[0]] = 1
		f.close()
		return res

	# get the bigram word's tf
	def get_tf(self, text):
		words = text.strip().split()
		word_num = len(words)
		word_cnt = {}
		feature = {}
		total = len(words)
		for i in range(total-1):
			biword = words[i]+words[i+1]
			tmp = word_cnt.get(biword, 0)
			word_cnt[biword] = tmp + 1
		for i in word_cnt:
			if i in self.feature_word:
				feature[i] = float(word_cnt[i])
		return feature

# online perceptron class
class OnlinePerceptron:

	# init perceptron
	def __init__(self):
		self.word_map = self.get_word_map()
		self.weights = [0.0 for _ in range(len(self.word_map))]
		self.bias = 0.0

	# show the weights
	def __str__(self):
		res_str = ''
		for i in self.final:
			res_str += '%f\t'%(i)
		res_str += '%f\n'%(self.final_bias)
		return res_str

	# init training turn 
	def init_turn(self):
		self.train_example = 1000000.0
		self.final = [0.0 for _ in range(len(self.word_map))]
		self.final_turn = [0.0 for _ in range(len(self.word_map))]
		self.bias_turn = 0.0
		self.final_bias = 0.0

	# update w_final
	def update_final(self, input_vec, turn):
		for i in input_vec:
			word = self.word_map[i]
			self.final[word] += (turn-self.final_turn[word])*self.weights[word]/self.train_example
			self.final_turn[word] = turn
		self.final_bias += (turn-self.bias_turn)*self.bias/self.train_example
		self.bias_turn = turn

	# def update_final_all(self, turn):
	# 	for i in range(len(self.final)):
	# 		self.final[i] += (turn-self.final_turn[i])/self.train_example
	# 		self.final_turn[i] = turn
	# 	self.final_bias += (turn-self.bias_turn)/self.train_example
	# 	self.bias_turn = turn

	# get the word list and save the map of word and number 
	def get_word_map(self):
		# f = open('word', 'r')      # used in unigram
		# f = open('biword', 'r')  # used in bigram
		f = open('idfword', 'r') # used in idfword
		res = {}
		cnt = 0
		for line in f.readlines():
			line = line.strip().split()
			res[line[0]] = cnt 
			cnt += 1
		f.close()
		return res

	# use the wi to predict result
	def predict(self, input_vec):
		res = 0.0
		for i in input_vec:
			res += self.weights[self.word_map[i]]*input_vec[i]
		res += self.bias
		return self.activator(res)

	# use the w_final to predict result
	def predict_final(self, input_vec):
		res = 0.0
		for i in input_vec:
			res += self.final[self.word_map[i]]*input_vec[i]
		res += self.final_bias
		return self.activator(res)

	# train the model
	def train(self, input_vecs, lable, rate):
		data = zip(input_vecs, lable)
		for i in data:
			output = self.predict(i[0])
			self.update_weight(i[0], output, i[1], rate)

	# update w_i
	def update_weight(self, input_vec, output, label, rate):
		delta = int(label) - output
		for i in input_vec:
			self.weights[self.word_map[i]] = self.weights[self.word_map[i]] + delta * rate * input_vec[i]
		self.bias += rate * delta

	# get the label
	def activator(self, x):
		return 1 if x > 0 else 0

	def getbiggest(self):
		tmp = {}
		for i in self.word_map:
			tmp[i] = self.final[self.word_map[i]]
		sort_weight = sorted(tmp.items(), key = lambda x:x[1], reverse = True)
		res = []
		for i in range(10):
			res.append(sort_weight[i][0])
		return res

	def getsmallest(self):
		tmp = {}
		for i in self.word_map:
			tmp[i] = self.final[self.word_map[i]]
		sort_weight = sorted(tmp.items(), key = lambda x:x[1], reverse = False)
		res = []
		for i in range(10):
			res.append(sort_weight[i][0])
		return res