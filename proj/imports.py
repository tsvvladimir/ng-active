from __future__ import division
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from numpy.random import randint
import logging
import sys
import operator
import math
from sklearn.multiclass import OneVsRestClassifier
from collections import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import wordnet as wn
import re
from itertools import chain
from sklearn.utils import shuffle
import nltk
import string
import os
from nltk.stem.porter import PorterStemmer
from string import maketrans