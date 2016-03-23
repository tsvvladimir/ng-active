from setup_data import *

def minimum_margin_init():
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', LinearSVC())
    ])
    #range by minimum margin
    print "range by minimum margin init"
    alpha = 50 #initial training set
    betha = 30 #number of iteration
    gamma = 50 #number of sampling

    #twenty_cur_training_data = twenty_train_data[:alpha]
    #twenty_cur_training_target = twenty_train_target[:alpha]
    #twenty_unlabeled_data = twenty_train_data[alpha:]
    #twenty_unlabeled_target = twenty_train_target[alpha:]

    #get decision function for entire train_data

    text_clf.fit(twenty_train_data, twenty_train_target)
    confidence_scores_all = text_clf.decision_function(twenty_train_data)
    doc_score_all = {}
    for i in range(0, len(twenty_train_data)):
        #print confidence_scores_all.shape
        last_elems = (sorted(confidence_scores_all[i]))[-2:]
        doc_score_all[i] = np.abs(last_elems[0] - last_elems[1])
    sorted_doc_score_all = sorted(doc_score_all.items(), key=operator.itemgetter(1))

    sample_numbers_all = np.array([])
    for i in range(-alpha, 0):
        sample_numbers_all = np.append(sample_numbers_all, sorted_doc_score_all[i][0])
    #sample_numbers_all - for forming cur_training_data and cur_training_target

    twenty_cur_training_data = []
    twenty_cur_training_target = []
    sample_data_all = list(twenty_train_data)
    sample_target_all = list(twenty_train_target)
    for i in range(0, len(sample_numbers_all)):
        temp1 = twenty_train_data[int(sample_numbers_all[i])]
        temp2 = twenty_train_target[int(sample_numbers_all[i])]
        twenty_cur_training_data = np.append(twenty_cur_training_data, temp1)
        twenty_cur_training_target = np.append(twenty_cur_training_target, temp2)
        sample_data_all.pop(i)
        sample_target_all.pop(i)
    twenty_unlabeled_data = sample_data_all
    twenty_unlabeled_target = sample_target_all

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

