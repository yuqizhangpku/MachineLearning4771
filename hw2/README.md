This question asked to work on text classification based on Online Perceptron. The training set given is 580M while test set is 190M.
The data is from Yelp, which provides the text(normalized) and corresponding label(1 as ranked more than 4 while 0 as not)

Here I split the whole training and prediction process into 4 part: 


(a) In classfile.py I defined the calculation of tf, tf-idf, tf of bigram and online-perceptron.

(b) In count bigram/idf/unigram.py, I input the original data set and save the frequency of each word
as file idfword/biword/word. Here in each model, I delete the words from stopWords and the ones that appear only once, which
do not provide enough information.

(c) In count accuracy bigram/idf.py, I use the perceptron defined in classfile.py and clean data from count bigram/idf/unigram.py to training the corresponding w and record it. The most impressive
10 wi is also recorded to answer question (iii).

(d) In main bigram/idf.py, I select a part of data from training set (16, 32, 64, ... , 2^n) to compare
the accuracy and give corresponding sketch.
