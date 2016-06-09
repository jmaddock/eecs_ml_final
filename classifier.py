# coding: utf-8

from collections import OrderedDict
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import KFold
from sklearn import metrics
import pandas as pd
import json
import pickle
import rumor_terms
import re
import numpy as np
import random
import os
import argparse

#vectorizor info
analyzer = 'char_wb'
ngram_range = (1,5)
stopwords = 'english'
tfidf = False

# extract the text column from a DataFrame
class textExtractor(TransformerMixin):
    def transform(self, X, **transform_params):
        return X['text'].values
    def fit(self, X, y=None, **fit_params):
        return self

# extract the features column from a DataFrame
class questionExtractor(TransformerMixin):
    def transform(self, X, **transform_params):
        return X['is_question'].values.reshape(len(X),1)
    def fit(self, X, y=None, **fit_params):
        return self

class mentionExtractor(TransformerMixin):
    def transform(self, X, **transform_params):
        return X['has_mention'].values.reshape(len(X),1)
    def fit(self, X, y=None, **fit_params):
        return self

class urlExtractor(TransformerMixin):
    def transform(self, X, **transform_params):
        return X['has_url'].values.reshape(len(X),1)
    def fit(self, X, y=None, **fit_params):
        return self

class hashtagExtractor(TransformerMixin):
    def transform(self, X, **transform_params):
        return X['has_hashtag'].values.reshape(len(X),1)
    def fit(self, X, y=None, **fit_params):
        return self
    
# make a featureset and train a classifier
def train_classifier(labled_data,classifier_type,features_to_use):
    if classifier_type == 'max_ent':
        classifier = LogisticRegression()
    elif classifier_type == 'nb':
        classifier = MultinomialNB()
    elif classifier_type == 'svm':
        classifier = SVC()
    elif classifier_type == 'knn':
        classifier = KNeighborsClassifier()
    elif classifier_type == 'dtree':
        classifier = DecisionTreeClassifier()
    elif classifier_type == 'random_forest':
        classifier = RandomForestClassifier()

    feature_list = [
        ('bag_of_words',Pipeline([
            ('extractor',textExtractor()),
            ('vectorizer',CountVectorizer(analyzer='char_wb',
                                          ngram_range=(1,5),
                                          stop_words=None)),
            ('transformer',TfidfTransformer(use_idf=False))
        ])),
    ]
    if 'uncertainty_terms' in features_to_use or 'all' in features_to_use:
        feature_list.append(
            ('uncertainty_terms',Pipeline([
                ('extractor',textExtractor()),
                ('vectorizer',CountVectorizer(analyzer='word',
                                              ngram_range=(1,1),
                                              stop_words=None,
                                              vocabulary=rumor_terms.stemmed)),
                ('transformer',TfidfTransformer(use_idf=False))
            ])))
    if 'question' in features_to_use or 'all' in features_to_use:
        feature_list.append(
            ('question_feature',Pipeline([
                ('extractor',questionExtractor()),
                #('vectorizor',CountVectorizer())
            ])))
    if 'mention' in features_to_use or 'all' in features_to_use:
        feature_list.append(
            ('mention_feature',Pipeline([
                ('extractor',mentionExtractor()),
            ])))
    if 'hashtag' in features_to_use or 'all' in features_to_use:
        feature_list.append(
            ('hashtag_feature',Pipeline([
                ('extractor',hashtagExtractor()),
            ])))
    if 'url' in features_to_use or 'all' in features_to_use:
        feature_list.append(
            ('url_feature',Pipeline([
                ('extractor',urlExtractor()),
            ])))
    pipeline = Pipeline([
        ('features',FeatureUnion(feature_list)),
        ('classifier',classifier)
    ])
    pipeline.fit(labled_data,
                 labled_data['class'].values)
    return pipeline

# validate the classifier over zipped training and testing datasets
# can be a single train/test pair or multiple zipped together

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create uncertainty classifier')
    parser.add_argument('-i','--infile')
    parser.add_argument('-o','--outfile')
    parser.add_argument('-c','--classifier_type',nargs='2')
    parser.add_argument('-f','--features',nargs='*')
    parser.add_argument('--balanced',action='store_true')
    parser.add_argument('--verbose',action='store_true')
    args = parser.parse_args()
    df = pd.read_csv(args.infile,dtype={'text': object})
    df = df.loc[df['text'].notnull()]
    k_fold_validation(df,args.classifier_type,args.features,args.verbose)

