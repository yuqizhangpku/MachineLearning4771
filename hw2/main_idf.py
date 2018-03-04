from count_accuracy_idf import count_accuracy
from count_idf import count_idf
import matplotlib.pyplot as plt

def plant_idf():
	f = open('idf_unigram_result', 'r')
	x = []
	y = []
	for line in f.readlines():
		line = line.strip().split('\t')
		x.append(int(line[0]))
		y.append(float(line[1]))
	plt.plot(x,y)
	plt.show()

def count():
	train_example = 2**5
	f = open('idf_unigram_result', 'w')
	while(train_example < 1000000):
		count_idf(train_example)
		res = count_accuracy(train_example)
		f.write('%s\t%s\n'%(str(train_example), str(res)))
		train_example *= 2

if __name__ == "__main__":
	count()
	plant_idf()