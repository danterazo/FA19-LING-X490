# LING-X 490 Assignment 7: Kumar Random Forest
# Dante Razo, drazo, 11/21/2019
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sklearn.metrics
import pandas as pd
import random
