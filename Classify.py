# -*- coding: utf-8 -*-
"""
Acknowledgements to @author: vsnick
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
train = pd.read_csv('train2.csv')
train.symptoms
train.disease
# Vectorizing the symptoms
tfid_vectorizer = TfidfVectorizer(min_df=1)
X = tfid_vectorizer.fit_transform(train.symptoms)
tfid_vectorizer.get_feature_names()

# Vectorizing the diseases
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Y = le.fit_transform(train.disease)
le.classes_

# implement Multinomial NB
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X,Y)

test = pd.read_csv('test2.csv')
test.symptoms

predicted = clf.predict_proba(tfid_vectorizer.transform(test.symptoms))
prob = predicted.tolist()[0]
prob = [x*100 for x in prob]
# Probabilities of diseases
output = dict(zip(le.classes_,prob))
print pd.DataFrame(output.items())
# Most probable
predicted = clf.predict(tfid_vectorizer.transform(test.symptoms))
print le.classes_[predicted]

# implement Binomial NB
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB().fit(X,Y)

test = pd.read_csv('test2.csv')
test.symptoms

predicted = clf.predict_proba(tfid_vectorizer.transform(test.symptoms))
prob = predicted.tolist()[0]
prob = [x*100 for x in prob]
# Probabilities of diseases
output = dict(zip(le.classes_,prob))
print pd.DataFrame(output.items())
# Most probable
predicted = clf.predict(tfid_vectorizer.transform(test.symptoms))
print le.classes_[predicted]



