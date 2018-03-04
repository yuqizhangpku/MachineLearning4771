from count_accuracy_bigram import count_accuracy
from count_bigram import count_bigram
import matplotlib.pyplot as plt

def plant():
	f = open('idf_unigram_result', 'r')
	x = []
	y = []
	for line in f.readlines():
		line = line.strip().split('\t')
		x.append(int(line[0]))
		y.append(1-float(line[1]))
	plt.plot(x,y)
	plt.show()

def count():
	train_example = 2**5
	f = open('bigram_result', 'w')
	while(train_example < 1000000):
		count_bigram(train_example)
		res = count_accuracy(train_example)
		f.write('%s\t%s\n'%(str(train_example), str(res)))
		train_example *= 2

if __name__ == "__main__":
	count()
	plant()