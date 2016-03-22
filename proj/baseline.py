from setup_data import *

def baseline():

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', LinearSVC())
    ])

    #baseline

    text_clf.fit(twenty_train_data, twenty_train_target)
    predicted = text_clf.predict(twenty_test_data)
    cur_score = f1_score(twenty_test_target, predicted, average='macro')
    print "baseline"
    print "(", len(twenty_test_target), ", ", cur_score, ")"

    #count documents most similar to theme's name
    #twenty_train_data and twenty_train_target
    #theme names twenty_test.target_names

