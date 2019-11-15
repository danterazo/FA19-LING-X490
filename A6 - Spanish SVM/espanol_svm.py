# LING-X 490 Assignment 6: Spanish SVM
# Dante Razo, drazo, 11/14/2019
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.metrics
import numpy as np
import random

# Import data
# TODO: idea: remove http://t.co/* links

# go to paper, see how they split kaggle dataset and split accordingly
# let sandra know; we want the same data

a.close()

# Feature engineering: vectorizer
# ML models need features, not just whole tweets
print("COUNTVECTORIZER CONFIG\n----------------------")
analyzer = input("Please enter analyzer: ")
ngram_upper_bound = input("Please enter ngram_upper_bound: ")

vec = CountVectorizer(analyzer=analyzer, ngram_range=(1, int(ngram_upper_bound)))  # TODO: word vs char, ngram_range
print("\nFitting CV...")
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)

# Shuffle data (keeps indices)
X_train, y_train = shuffle(X_train, y_train)
X_test, y_test = shuffle(X_test, y_test)

# Fitting the model
print("Training SVM...")
svm = SVC(kernel="linear", gamma="auto")  # TODO: tweak params
svm.fit(X_train, y_train)
print("Training complete.\n")

""" KERNEL RESULTS gamma="auto", analyzer=word, ngram_range(1,3)
linear: 
rbf: 
poly: 
sigmoid: 
precomputed: N/A, not supported
"""

# Testing + results
rand_acc = sklearn.metrics.balanced_accuracy_score(y_test, [random.randint(1, 2) for x in range(0, len(y_test))])
print(f"Random/Baseline Accuracy: {rand_acc}")
print(f"Testing Accuracy: {sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))}")

""" CV PARAM TESTING (kernel="linear")
word, ngram_range(1,2):  
word, ngram_range(1,3):  
word, ngram_range(1,5):  
word, ngram_range(1,10): 
char, ngram_range(1,2):  
char, ngram_range(1,3):  
char, ngram_range(1,5):  
char, ngram_range(1,10): 
"""
