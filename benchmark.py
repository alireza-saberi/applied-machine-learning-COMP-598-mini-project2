#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division # force floating point division
from __future__ import print_function
import csv
import logging
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import re
import snowballstemmer
import sys

from nltk.corpus import stopwords
from optparse import OptionParser
from sklearn import preprocessing
from sklearn.datasets import fetch_20newsgroups
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from time import time

### PARAMETERS
TRAIN_INPUT_FILE = 'data/train_input.csv'
TRAIN_OUTPUT_FILE = 'data/train_output.csv'
TEST_INPUT_FILE = 'data/test_input.csv'

category_data = list(csv.DictReader(open(TRAIN_OUTPUT_FILE), delimiter=',', quotechar='"'))
categories = np.array([item['category'] for item in category_data])

uniq_categories = list(set(categories))
le = preprocessing.LabelEncoder()
le.fit(uniq_categories)
Y = le.transform(categories)
print(uniq_categories)
print(le.transform(uniq_categories))

X = joblib.load('X_tfidf_trigrams.pkl')
#XX = joblib.load('XX_tfidf_trigrams.pkl')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print(X_train.shape)

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, Y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.f1_score(Y_test, pred)
    print("f1-score:   %0.3f" % score)

    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))
        print()

    print("classification report:")
    print(metrics.classification_report(Y_test, pred))

    print("confusion matrix:")
    print(metrics.confusion_matrix(Y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

results = []
for clf, name in (
#        (KNeighborsClassifier(n_neighbors=2), "kNN"),
        (RidgeClassifier(tol=1e-2, solver="lsqr", class_weight={0:0.95}), "Ridge Classifier"),
        (Perceptron(n_iter=50, class_weight="auto"), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50, n_jobs=4), "Passive-Aggressive")):
    print('=' * 80)
    print(name)
    result = benchmark(clf)
    results.append(result)
    print(result)

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty, dual=False, tol=1e-3, class_weight='auto')))

    # Train SGD model
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))


class L1LinearSVC(LinearSVC):
    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        self.transformer_ = LinearSVC(penalty="l1", dual=False, tol=1e-3, class_weight='auto')
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)

print('=' * 80)
print("LinearSVC with L1-based feature selection")
results.append(benchmark(L1LinearSVC()))

# make some plots
indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

clf_names, score, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
plt.barh(indices, score, .2, label="score", color='r')
plt.barh(indices + .3, training_time, .2, label="training time", color='g')
plt.barh(indices + .6, test_time, .2, label="test time", color='b')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.savefig('benchmark.jpg')
