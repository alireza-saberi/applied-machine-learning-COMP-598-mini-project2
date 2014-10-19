#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division # force floating point division
import csv
import math
import nltk
import numpy as np
import re
import snowballstemmer
import time

from contextlib import contextmanager
from nltk.corpus import stopwords
from random import randrange
from sklearn.cross_validation import KFold
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cross_validation import train_test_split

TRAIN_INPUT_FILE = 'data/train_input.csv'
TRAIN_OUTPUT_FILE = 'data/train_output.csv'
TEST_INPUT_FILE = 'data/test_input.csv'

abstract_data = list(csv.DictReader(open(TRAIN_INPUT_FILE), delimiter=',', quotechar='"'))
category_data = list(csv.DictReader(open(TRAIN_OUTPUT_FILE), delimiter=',', quotechar='"'))
titles = np.array([item['abstract'] for item in abstract_data])
categories = [item['category'] for item in category_data]

uniq_categories = list(set(categories))
le = preprocessing.LabelEncoder()
le.fit(uniq_categories)
Y = le.transform(categories)

### PARAMETERS
NUM_F = 2000
CSV_FILE = 'items.csv'
ENABLE_DEBUG = False
ENABLE_PROFILE = False
ENABLE_SAMPLING = True

### GLOBALS
stops = set(stopwords.words("english"))
stemmer = snowballstemmer.stemmer('english')

@contextmanager
def measureTime(title):
  if ENABLE_PROFILE:
    t1 = time.clock()
    yield
    t2 = time.clock()
    print '[%s] took %0.3f ms' % (title, (t2-t1)*1000.0)
  else:
    yield

class NaiveBayes:
  def __init__(self):
    self.max_features = None
    self.H = {}
    self.vocabulary = {}

  def set_max_features(self, max_features):
    self.max_features = max_features
    self.top_tokens = sorted(self.vocabulary, key=self.vocabulary.get, reverse=True)[:self.max_features]
  
  def fit(self, X_train, X_target):
    if ENABLE_SAMPLING:
      titles_new = []
      categories_new = []
      count_by_category = {}
      titles_by_category = {}
      for i, title in enumerate(X_train):
        category = X_target[i]
        if not category in count_by_category:
          count_by_category[category] = 0
          titles_by_category[category] = []
        count_by_category[category] += 1
        titles_by_category[category].append(title)
      for w in sorted(count_by_category, key=count_by_category.get, reverse=True):
        avg = int(len(X_train) / len(count_by_category))
        if count_by_category[w] <= avg:
          for i in xrange(avg):
            idx = randrange(len(titles_by_category[w]))
            titles_new.append(titles_by_category[w][idx])
            categories_new.append(w)
        else:
          for i in xrange(avg):
            idx = randrange(len(titles_by_category[w]))
            categories_new.append(w)
            titles_new.append(titles_by_category[w][idx])
      X_train = np.array(titles_new)
      X_target = np.array(categories_new)

    with measureTime("fit"):
      self.P = {}
      self.N = len(X_target)
      uniq_categories = np.unique(X_target)
      for category in uniq_categories:
        self.H[category] = { 'count': 0 }
        self.P[category] = 0
      for i, category in enumerate(X_target):
        self.H[category]['count'] += 1
        item = X_train[i]
        tokens = tokenize(item)
        for token in tokens:
          if not token in self.vocabulary:
            self.vocabulary[token] = 0
          self.vocabulary[token] += 1
        for token in set(tokens):
          if not token in self.H[category]:
            self.H[category][token] = 0
          self.H[category][token] += 1
      self.top_tokens = sorted(self.vocabulary, key=self.vocabulary.get, reverse=True)[:self.max_features]

      for category in uniq_categories:
        for token in self.top_tokens:
          count = self.H[category]['count']
          c1 = count
          if token in self.H[category]:
            c1 -= self.H[category][token]
          self.P[category] += math.log((c1+1)/(count+2))

      return self

  def predict_proba(self, tokens, category):
    if category is None:
      return np.array([self.predict_proba(tokens, category) for category in self.H])

    if category not in self.H: # category wasn't in training set ?!
      return 1 / (self.N+2)

    c = self.H[category]
    p = math.log((c['count']+1) / (self.N+2))
    p += self.P[category]

    for token in tokens:
      if token in self.top_tokens:
        c1 = 0
        if token in c:
          c1 = c[token]
        p += math.log((c['count']+2) / (c['count']-c1-1+2)) # undo addition in self.P
        p += math.log((c1+1) / (c['count']+2))
      else:
        p += math.log(1/(c['count']+2))
    return math.exp(p)

  def predict(self, X):
    pred = np.zeros((X.shape[0], 1))
    for i, x in enumerate(X):
      ps = self.predict_proba(tokenize(x), None)
      pred[i][0] = np.argmax(ps) # return index with highest probability
    return pred

  def score(self, X_test, Y_test):
    with measureTime("predictions"):
      predictions = [self.predict(x) for x in X_test]
    with measureTime("calculate accuracy"):
      ncorrect = 0
      ncorrect_by_category = {}
      count_by_category = {}
      for i in np.unique(Y_test):
        ncorrect_by_category[i] = 0
        count_by_category[i] = 0
      for i, category in enumerate(predictions):
        expected_category = Y_test[i]
        count_by_category[expected_category] += 1
        if category == expected_category:
          ncorrect += 1
          ncorrect_by_category[expected_category] += 1
      macro_avgs = [v/count_by_category[k] for k,v in ncorrect_by_category.iteritems()]
      d = {}
      d['micro'] = ncorrect / float(len(Y_test))
      d['macro'] = sum(macro_avgs) / float(len(macro_avgs))
      return d

class MultiNaiveBayes:
  def __init__(self):
    self.max_features = None
    self.H = {}
    self.vocabulary = {}
  
  def fit(self, X_train, X_target):
    if ENABLE_SAMPLING:
      titles_new = []
      categories_new = []
      count_by_category = {}
      titles_by_category = {}
      for i, title in enumerate(X_train):
        category = X_target[i]
        if not category in count_by_category:
          count_by_category[category] = 0
          titles_by_category[category] = []
        count_by_category[category] += 1
        titles_by_category[category].append(title)
      for w in sorted(count_by_category, key=count_by_category.get, reverse=True):
        avg = int(len(X_train) / len(count_by_category))
        if count_by_category[w] <= avg:
          for i in xrange(avg):
            idx = randrange(len(titles_by_category[w]))
            titles_new.append(titles_by_category[w][idx])
            categories_new.append(w)
        else:
          for i in xrange(avg):
            idx = randrange(len(titles_by_category[w]))
            categories_new.append(w)
            titles_new.append(titles_by_category[w][idx])
      X_train = np.array(titles_new)
      X_target = np.array(categories_new)

    with measureTime("fit"):
      self.N = len(X_target)
      for category in np.unique(X_target):
        self.H[category] = { 'count': 0, 'token_count': 0 }
      for i, category in enumerate(X_target):
        self.H[category]['count'] += 1
        item = X_train[i]
        tokens = tokenize(item)
        for token in tokens:
          if not token in self.vocabulary:
            self.vocabulary[token] = 0
          self.vocabulary[token] += 1
          self.H[category]['token_count'] += 1
          if not token in self.H[category]:
            self.H[category][token] = 0
          self.H[category][token] += 1
      return self

  def set_max_features(self, max_features):
    self.max_features = max_features
    self.top_tokens = sorted(self.vocabulary, key=self.vocabulary.get, reverse=True)[:self.max_features]

  def predict_proba(self, tokens, category):
    if category is None:
      return [self.predict_proba(tokens, category) for category in self.H]

    if category not in self.H: # category wasn't in training set ?!
      return 1 / (self.N+2)

    c = self.H[category]
    p = math.log((c['count']+1) / (self.N+2))

    for token in tokens:
      count = 0
      if token in c and token in self.top_tokens:
        count = c[token]
      p += math.log((count+1) / (c['token_count']+len(self.vocabulary)))
    return math.exp(p)

  def predict(self, X):
    pred = np.zeros((X.shape[0], 1))
    for i, x in enumerate(X):
      ps = self.predict_proba(tokenize(x), None)
      pred[i][0] = ps.index(max(ps)) # return index with highest probability
    return pred

  def score(self, X_test, Y_test):
    with measureTime("predictions"):
      predictions = [self.predict(x) for x in X_test]
    with measureTime("calculate accuracy"):
      ncorrect = 0
      ncorrect_by_category = {}
      count_by_category = {}
      for i in np.unique(Y_test):
        ncorrect_by_category[i] = 0
        count_by_category[i] = 0
      for i, category in enumerate(predictions):
        expected_category = Y_test[i]
        count_by_category[expected_category] += 1
        if category == expected_category:
          ncorrect += 1
          ncorrect_by_category[expected_category] += 1
      macro_avgs = [v/count_by_category[k] for k,v in ncorrect_by_category.iteritems()]
      d = {}
      d['micro'] = ncorrect / float(len(Y_test))
      d['macro'] = sum(macro_avgs) / float(len(macro_avgs))
      return d

def print_stats(items_data):
  count_by_category = {}
  print "Total items: %d" % len(items_data)
  for item in items_data:
    title = item['title']
    category = item['primary_category']
    if not category in count_by_category:
      count_by_category[category] = 0
    count_by_category[category] += 1
  for w in sorted(count_by_category, key=count_by_category.get, reverse=True):
    print w, count_by_category[w]
#  print sorted(count_by_category.keys())

def tokenize(s):
  s = s.lower()
  tokens = re.split('( |\/|\.|,|;|:|\(|\)|"|\?|!|®|ᴹᴰ|™|\*|\{|\}|\$|_|\\\\)', s)
  tokens = filter(lambda x: len(x) > 2 and x not in stops, tokens)
  return stemmer.stemWords(tokens)

X_train, X_test, Y_train, Y_test = train_test_split(titles, Y, test_size=0.2, random_state=42)
clf = NaiveBayes()
print "HERE"
clf.fit(X_train, Y_train)
#joblib.dump(clf, 'multi_bayes.pkl', compress=9)
print "done fit"
#clf = joblib.load('multi_bayes.pkl')
print "HIYA"
for k in xrange(500, 10000, 500):
  clf.set_max_features(k)
  print k
  pred = clf.predict(X_test)
  score = metrics.f1_score(Y_test, pred)
  print "NUM_F: %d" % k
  print metrics.classification_report(Y_test, pred)
  print metrics.confusion_matrix(Y_test, pred) 
