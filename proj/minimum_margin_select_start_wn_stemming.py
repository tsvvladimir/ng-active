from setup_data import *

def minimum_margin_select_start_wn_stemming():

    stemmer = PorterStemmer()

    def stem_tokens(tokens, stemmer):
        stemmed = []
        for item in tokens:
            stemmed.append(stemmer.stem(item))
        return stemmed

    def tokenize(text):
        tokens = nltk.word_tokenize(text)
        stems = stem_tokens(tokens, stemmer)
        return stems

    text_clf = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
        ('tfidf',TfidfTransformer()),
        ('clf', LinearSVC())
    ])
    #range by minimum margin
    print "range by minimum margin select start wordnet and stemmer"
    alpha = 100 #initial training set
    betha = 15 #number of iteration
    gamma = 50 #number of sampling

    #transform twenty_train_data and twenty_test_data

    transformed_twenty_train_data = []
    transformed_twenty_test_data = []

    replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation)).decode("latin-1")

    #print replace_punctuation


    for item in twenty_train_data:
        lowers = item.lower()
        no_punctuation = lowers.translate(replace_punctuation)
        transformed_twenty_train_data.append(no_punctuation)

    for item in twenty_test_data:
        lowers = item.lower()
        no_punctuation = lowers.translate(replace_punctuation)
        transformed_twenty_test_data.append(no_punctuation)

    help_conv = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, stop_words='english')),
        ('tfidf', TfidfTransformer())
    ])

    aug_themes = twenty_test.target_names[:]

    for i in range(0, len(aug_themes)):
        theme = aug_themes[i]
        #print 'theme:', theme
        theme_words = re.split(r"[\.\-]", theme)
        #add stop words
        for word in theme_words:
            syns = wn.synsets(word)
            sns = list(set(chain.from_iterable([word.lemma_names() for word in syns])))
            sns2 = list(set(chain.from_iterable([word.definition() for word in syns])))
            sns3 = list(set(chain.from_iterable([word.examples() for word in syns])))
            theme_words = theme_words + sns + sns2 + sns3
        #print theme_words
        aug_themes[i] = " ".join(theme_words)
    aug = aug_themes + transformed_twenty_train_data
    aug = help_conv.fit_transform(aug)

    cos_sims = []
    for i in range(0, 20):
        cos_sims.append(linear_kernel(aug[i:i+1], aug).flatten())
    rel_docs_idx = []
    for i in range(0, 20):
        rel_docs_idx.append(reversed(cos_sims[i].argsort()))

    docs_idx_take = []
    for i in range(0, 20):
        docs_idx_take.append(filter(lambda x: x >= 20, rel_docs_idx[i])[0:10])

    print docs_idx_take

    trues = {}
    for i in range(0, 20):
        for item in docs_idx_take[i]:
            #print twenty_train_data[item], twenty_train_target[item] == i
            if twenty_train_target[item] == i:
                trues[i] = trues.get(i, 0) + 1

    print len(trues)




    twenty_cur_training_data = []
    twenty_cur_training_target = []
    twenty_unlabeled_data = []
    twenty_unlabeled_target = []

    for lst in docs_idx_take:
        for idx in lst:
            twenty_cur_training_data.append(transformed_twenty_train_data[idx])
            twenty_cur_training_target.append(twenty_train_target[idx])

    for i in range(0, len(transformed_twenty_train_data)):
        if i not in [item for sublist in docs_idx_take for item in sublist]:
            twenty_unlabeled_data.append(transformed_twenty_train_data[i])
            twenty_unlabeled_target.append(twenty_train_target[i])

    #twenty_cur_training_data = twenty_train_data[:alpha]
    #twenty_cur_training_target = twenty_train_target[:alpha]
    #twenty_unlabeled_data = twenty_train_data[alpha:]
    #twenty_unlabeled_target = twenty_train_target[alpha:]






    text_clf.fit(twenty_cur_training_data, twenty_cur_training_target)
    predicted = text_clf.predict(transformed_twenty_test_data)
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
        predicted = text_clf.predict(transformed_twenty_test_data)
        cur_score = f1_score(twenty_test_target, predicted, average='macro')
        print "(", len(twenty_cur_training_data), "; ", cur_score, ")"

