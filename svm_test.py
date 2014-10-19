#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division # force floating point division
from __future__ import print_function
import csv
import math
import nltk
import numpy as np
import re
import snowballstemmer
import sys
import time

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nolearn.dbn import DBN
from optparse import OptionParser
from random import randrange
from scipy.fftpack import *
from scipy.sparse import vstack
from sklearn import grid_search
from sklearn import metrics
from sklearn import preprocessing
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD, PCA, RandomizedPCA
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import RidgeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.extmath import density
from sklearn.cross_validation import train_test_split

### PARAMETERS
TRAIN_INPUT_FILE = 'data/train_input.csv'
TRAIN_OUTPUT_FILE = 'data/train_output.csv'
TEST_INPUT_FILE = 'data/test_input.csv'
CREATE_SUBMISSION = True
USE_CACHE = True

def get_stop_words():
  f = open('data/stopwords.txt')
  L = []
  for w in f:
    L.append(w.strip())
  return set(L)

def get_indices():
  f = open('data/uniq_indices.txt')
  L = set()
  for w in f:
    L.add(int(w.strip()))
  return list(L)
indices = get_indices()

### GLOBALS
stops = get_stop_words()
stemmer = snowballstemmer.stemmer('english')

abstract_data = list(csv.DictReader(open(TRAIN_INPUT_FILE), delimiter=',', quotechar='"'))
category_data = list(csv.DictReader(open(TRAIN_OUTPUT_FILE), delimiter=',', quotechar='"'))

abstracts_all = np.array([item['abstract'] for item in abstract_data])
categories_all = np.array([item['category'] for item in category_data])
abstracts = abstracts_all#[indices]
categories = categories_all#[indices]

def tokenize(s):
  s = s.lower()
  tokens = re.split('( |\/|\.|,|;|:|\(|\)|"|\?|!|®|ᴹᴰ|™|\*|\{|\}|\$|_|\\\\)', s)
  tokens = filter(lambda x: len(x) > 2 and x not in stops, tokens)
  return stemmer.stemWords(tokens)

test_data = list(csv.DictReader(open(TEST_INPUT_FILE), delimiter=',', quotechar='"'))
test_abstracts = np.array([item['abstract'] for item in test_data])

uniq_categories = list(set(categories.tolist()))
le = preprocessing.LabelEncoder()
le.fit(uniq_categories)
Y = le.transform(categories)
print(uniq_categories)
print(le.transform(uniq_categories))

X = np.array(abstracts).T
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_submit = np.array(test_abstracts).T

vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1,1))
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
X_submit = vectorizer.transform(X_submit)

print('done vectorizing')

def dump_csv(pred, k, epochs):
  f = open('output/preds_dbn_k%d_e%d.csv' % (k, epochs), 'w')
  f.write('"id","category"\n')
  for i, x in enumerate(pred):
    idx = test_data[i]['id']
    klass = le.inverse_transform([int(x)])[0]
    f.write('"%s","%s"\n' % (idx, klass))
  f.close()

def benchmark(k, epochs):
  print("*" * 80)
  print("k: %d, epochs: %d\n" % (k, epochs))

  #select = SelectKBest(score_func=chi2, k=k)
  select = TruncatedSVD(n_components=k)
  X_train_trunc = select.fit_transform(X_train, Y_train)
  X_test_trunc = select.transform(X_test)

  print('done truncating')

  parameters = {'C': [1, 10, 100, 1000, 10000],  'class_weight': ['auto', None], 'tol':[0.001,0.0001]}
  clf = LinearSVC(C=100000)
  #clf = grid_search.GridSearchCV(svc, parameters)
  clf.fit(X_train_trunc, Y_train)
  pred = clf.predict(X_test_trunc)

  if CREATE_SUBMISSION:
    X_submit_trunc = select.transform(X_submit)
    pred_submit = clf.predict(X_submit_trunc)
    dump_csv(pred_submit, k, epochs)

  score = metrics.f1_score(Y_test, pred)
  print("f1-score:   %0.3f" % score)

  print("classification report:")
  print(metrics.classification_report(Y_test, pred))

  print("confusion matrix:")
  print(metrics.confusion_matrix(Y_test, pred))

benchmark(100, 20)
