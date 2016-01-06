from imports import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

twenty_train = fetch_20newsgroups(subset='train')
twenty_train_data = twenty_train.data
twenty_train_target = twenty_train.target
twenty_test = fetch_20newsgroups(subset='test')
twenty_test_data = twenty_test.data
twenty_test_target = twenty_test.target