from setup_data import *

def random_sampling():
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', LinearSVC())
    ])

    #add random elements strategy
    print "random sampling solution"
    alpha = 100 #initial training set
    betha = 140 #number of iteration
    gamma = 50 #number of sampling

    twenty_cur_training_data = twenty_train_data[:alpha]
    twenty_cur_training_target = twenty_train_target[:alpha]
    twenty_unlabeled_data = twenty_train_data[alpha:]
    twenty_unlabeled_target = twenty_train_target[alpha:]

    text_clf.fit(twenty_cur_training_data, twenty_cur_training_target)
    predicted = text_clf.predict(twenty_test_data)
    cur_score = f1_score(twenty_test_target, predicted, average='micro')
    print "(", len(twenty_cur_training_data), ", ", cur_score, ")"

    for t in range(1, betha):
        sample_numbers = randint(0, len(twenty_unlabeled_data), gamma)
        sample_data = list(twenty_unlabeled_data)
        sample_target = list(twenty_unlabeled_target)
        for i in range(0, len(sample_numbers)):
            temp1 = twenty_unlabeled_data[sample_numbers[i]]
            temp2 = twenty_unlabeled_target[sample_numbers[i]]
            twenty_cur_training_data = np.append(twenty_cur_training_data, temp1)
            twenty_cur_training_target = np.append(twenty_cur_training_target, temp2)
            sample_data.pop(i)
            sample_target.pop(i)
        twenty_unlabeled_data = sample_data
        twenty_unlabeled_target = sample_target

        text_clf.fit(twenty_cur_training_data, twenty_cur_training_target)
        predicted = text_clf.predict(twenty_test_data)
        cur_score = f1_score(twenty_test_target, predicted, average='micro')
        print "(", len(twenty_cur_training_data), ", ", cur_score, ")"

