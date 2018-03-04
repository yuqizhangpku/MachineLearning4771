import matplotlib.pyplot as plt

def plant():
	f = open('unigram_result', 'r')
	x = []
	y = []
	for line in f.readlines():
		line = line.strip().split('\t')
		x.append(int(line[0]))
		y.append(1-float(line[1]))
	plt.plot(x,y,label='unigram')
	f = open('idf_unigram_result', 'r')
	x = []
	y = []
	for line in f.readlines():
		line = line.strip().split('\t')
		x.append(int(line[0]))
		y.append(1-float(line[1]))
	plt.plot(x,y,label='tf-idf')
	f = open('bigram_result', 'r')
	x = []
	y = []
	for line in f.readlines():
		line = line.strip().split('\t')
		x.append(int(line[0]))
		y.append(1-float(line[1]))
	plt.plot(x,y,label='bigram')
	plt.legend() 
	plt.ylim(0.75, 0.9)
	plt.show()

if __name__ == "__main__":
	plant()