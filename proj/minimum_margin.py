from setup_data import *

def minimum_margin():
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', LinearSVC())
    ])
    #range by minimum margin
    print "range by minimum margin"
    alpha = 100 #initial training set
    betha = 140 #number of iteration
    gamma = 50 #number of sampling

    twenty_cur_training_data = twenty_train_data[:alpha]
    twenty_cur_training_target = twenty_train_target[:alpha]
    twenty_unlabeled_data = twenty_train_data[alpha:]
    twenty_unlabeled_target = twenty_train_target[alpha:]

    text_clf.fit(twenty_cur_training_data, twenty_cur_training_target)
    predicted = text_clf.predict(twenty_test_data)
    cur_score = f1_score(twenty_test_target, predicted, average='macro')
    print "(", len(twenty_cur_training_data), "; ", cur_score, ")"

    for t in range(1, betha):
        #sample_numbers = randint(0, len(twenty_unlabeled_data), gamma)
        confidence_scores = text_clf.decision_function(twenty_unlabeled_data)
        #print len(twenty_unlabeled_data)
        #print confidence_scores.shape

        doc_score = {}
        for i in range(0, len(twenty_unlabeled_data)):
            #sort confidence_score[i], get 2 last elements, find absolute value of difference between the,
            #print confidence_scores.shape
            last_elems = (sorted(confidence_scores[i]))[-2:]
            doc_score[i] = np.abs(last_elems[0] - last_elems[1])

        sorted_doc_score = sorted(doc_score.items(), key=operator.itemgetter(1))

        sample_numbers = np.array([])
        for i in range(0, gamma):
            #print i, type(sorted_doc_score[i][0])
            sample_numbers = np.append(sample_numbers, (sorted_doc_score[i][0]))

        #sample_numbers - indices which we should add
        sample_data = list(twenty_unlabeled_data)
        sample_target = list(twenty_unlabeled_target)
        for i in range(0, len(sample_numbers)):
            #print type(sample_numbers[i])
            temp1 = twenty_unlabeled_data[int(sample_numbers[i])]
            temp2 = twenty_unlabeled_target[int(sample_numbers[i])]
            twenty_cur_training_data = np.append(twenty_cur_training_data, temp1)
            twenty_cur_training_target = np.append(twenty_cur_training_target, temp2)
            sample_data.pop(i)
            sample_target.pop(i)
        twenty_unlabeled_data = sample_data
        twenty_unlabeled_target = sample_target

        text_clf.fit(twenty_cur_training_data, twenty_cur_training_target)
        predicted = text_clf.predict(twenty_test_data)
        cur_score = f1_score(twenty_test_target, predicted, average='macro')
        print "(", len(twenty_cur_training_data), "; ", cur_score, ")"

