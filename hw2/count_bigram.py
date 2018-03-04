from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))

def count_bigram(train_example):
	f = open('../reviews_tr.csv', 'r')
	lines = f.readlines()
	word_set = {}
	cnt = 0 
	total = len(lines)
	for line in lines[1:train_example]:
		text = line.strip().split(',')[1]
		words = text.split()
		word_total = len(words)
		for i in range(word_total-1):
			if words[i] not in stopWords:
				biword = words[i] + words[i+1]
				tmp = word_set.get(biword, 0)
				word_set[biword] = tmp + 1
		if cnt % 100 == 0:
			print(float(cnt)/total)
		cnt += 1
	f.close()

	f = open('biword', 'w')
	# print word_set
	sortword = sorted(word_set.items(),key = lambda x:x[1],reverse = True)
	for word in sortword:
		if word[1] > 1:
			f.write('%s\t%s\n'%(word[0], str(word[1])))
	f.close()

if __name__ == '__main__':
	count_bigram(1000000)