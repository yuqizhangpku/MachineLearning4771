from count_accuracy import count_accuracy
from count_unigram import count_unigram
import matplotlib.pyplot as plt

def plant_uni():
	f = open('unigram_result', 'r')
	x = []
	y = []
	for line in f.readlines():
		line = line.strip().split('\t')
		x.append(int(line[0]))
		y.append(float(line[1]))
	plt.plot(x,y)

def count():
	train_example = 2**5
	f = open('idf_unigram_result', 'w')
	while(train_example < 1000000):
		count_unigram(train_example)
		res = count_accuracy(train_example)
		f.write('%s\t%s\n'%(str(train_example), str(res)))
		train_example *= 2

if __name__ == "__main__":
	count()
	plant_uni()
