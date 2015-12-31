import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from numpy.random import randint
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

twenty_train = fetch_20newsgroups(subset='train')
twenty_train_data = twenty_train.data
twenty_train_target = twenty_train.target
twenty_test = fetch_20newsgroups(subset='test')
twenty_test_data = twenty_test.data
twenty_test_target = twenty_test.target

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('clf', SGDClassifier())
])
'''
#baseline

text_clf.fit(twenty_train_data, twenty_train_target)
predicted = text_clf.predict(twenty_test_data)
#print "baseline f1 micro", f1_score(twenty_test_target, predicted, average='micro')
'''
print "baseline f1 micro 0.849309612321"

#add random elements strategy
alpha = 100 #initial training set
betha = 100 #number of iteration
gamma = 50 #number of sampling

twenty_cur_training_data = twenty_train_data[:alpha]
twenty_cur_training_target = twenty_train_target[:alpha]
twenty_unlabeled_data = twenty_train_data[alpha:]
twenty_unlabeled_target = twenty_train_target[alpha:]

text_clf.fit(twenty_cur_training_data, twenty_cur_training_target)
predicted = text_clf.predict(twenty_test_data)
cur_score = f1_score(twenty_test_target, predicted, average='micro')
print "train set", len(twenty_cur_training_data), "cur_score", cur_score

for t in range(1, betha):
    sample_numbers = randint(0, len(twenty_unlabeled_data), gamma)
    sample_data = list(twenty_unlabeled_data)
    sample_target = list(twenty_unlabeled_target)
    #print "will take this idx", sample_numbers, "while length of set is", len(twenty_unlabeled_data)
    for i in range(0, len(sample_numbers)):
        #try:
        temp1 = twenty_unlabeled_data[sample_numbers[i]]
            #print "add document", temp1
        temp2 = twenty_unlabeled_target[sample_numbers[i]]
            #print "add target", temp2
        twenty_cur_training_data = np.append(twenty_cur_training_data, temp1)
        #print "1smth with idx", i
        twenty_cur_training_target = np.append(twenty_cur_training_target, temp2)
        #print "2smth with idx", i
        sample_data.pop(i)
        #print "3smth with idx", i
        sample_target.pop(i)
        #print "4smth with idx", i
        #except:
        #    print "Unexpected error:", sys.exc_info()[0]
    twenty_unlabeled_data = sample_data
    twenty_unlabeled_target = sample_target

    text_clf.fit(twenty_cur_training_data, twenty_cur_training_target)
    predicted = text_clf.predict(twenty_test_data)
    cur_score = f1_score(twenty_test_target, predicted, average='micro')
    print "train set", len(twenty_cur_training_data), "cur_score", cur_score