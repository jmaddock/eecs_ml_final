from collections import OrderedDict
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import *
from sklearn.neighbors import *
from sklearn.svm import *
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
from sklearn.learning_curve import learning_curve
import pandas as pd
import json
import pickle
import rumor_terms
import re
import numpy as np
import random
import os
import argparse
import classifier as Classifier

## VALIDATION PARAMS
KFOLDS = 10
KFOLD_PICKLED_FILE_PATH = '/Users/klogg/research_data/ml_uncertainty/kfold_splits.pickle'

def k_fold_validation(labled_data,classifier_type,features_to_use,verbose=True,outfile=None):
    scores = OrderedDict()
    scores['f1'] = []
    scores['recall'] = []
    scores['precision'] = []
    confusion = np.array([[0, 0], [0, 0]])
    train_and_test = kfold_split(labled_data,KFOLDS,KFOLD_PICKLED_FILE_PATH)
    for x,y in train_and_test:
        train_data = labled_data.iloc[x]
        test_data = labled_data.iloc[y]
        test_lables = labled_data.iloc[y]['class'].values

        classifier = Classifier.train_classifier(train_data,classifier_type,features_to_use)
        predictions = classifier.predict(test_data)
        
        confusion += metrics.confusion_matrix(test_lables, predictions)
        f1_score = metrics.f1_score(test_lables, predictions, pos_label=1)
        recall = metrics.recall_score(test_lables, predictions, pos_label=1)
        precision = metrics.precision_score(test_lables,predictions,pos_label=1)
        if verbose:
            print 'tweets classified:', len(y)
            print 'f1: %s' % f1_score
            print 'recall: %s' % recall
            print 'precision: %s\n' % precision
        scores['f1'].append(f1_score)
        scores['recall'].append(recall)
        scores['precision'].append(precision)

    print 'Classifier: {0}'.format(classifier_type)
    print 'Total tweets classified:', len(labled_data)
    for score in scores:
        scores[score] = sum(scores[score])/KFOLDS
    for score in scores:
        print '%s: %s' % (score,scores[score])
    print('Confusion matrix:')
    print(confusion)
    return pd.DataFrame([scores])

def rumor_validation(labled_data,classifier_type,features_to_use,verbose=True,outfile=None):
    scores = OrderedDict()
    scores['f1'] = []
    scores['recall'] = []
    scores['precision'] = []
    confusion = np.array([[0, 0], [0, 0]])
    train_and_test = rumor_split(labled_data)
    for x,y in train_and_test:
        train_data = labled_data.loc[x]
        test_data = labled_data.loc[y]
        test_lables = labled_data.loc[y]['class'].values

        classifier = Classifier.train_classifier(train_data,classifier_type,features_to_use)
        predictions = classifier.predict(test_data)
        
        confusion += metrics.confusion_matrix(test_lables, predictions)
        f1_score = metrics.f1_score(test_lables, predictions, pos_label=1)
        recall = metrics.recall_score(test_lables, predictions, pos_label=1)
        precision = metrics.precision_score(test_lables,predictions,pos_label=1)
        if verbose:
            print 'tweets classified:', len(y)
            print 'f1: %s' % f1_score
            print 'recall: %s' % recall
            print 'precision: %s\n' % precision
        scores['f1'].append(f1_score)
        scores['recall'].append(recall)
        scores['precision'].append(precision)

    print 'Classifier: {0}'.format(classifier_type)
    print 'Total tweets classified:', len(labled_data)
    for score in scores:
        scores[score] = sum(scores[score])/len(train_and_test)
    for score in scores:
        print '%s: %s' % (score,scores[score])
    print('Confusion matrix:')
    print(confusion)
    return pd.DataFrame([scores])


def kfold_split(labled_data,n_folds,fname=None):
    result = KFold(n=len(labled_data), n_folds=n_folds)
    if fname:
        f = open(fname, 'w')
        pickle.dump(result,f)
    return result

def rumor_split(labled_data):
    test_indices = []
    train_indices = []
    for rumor in df.rumor.unique():
        test_temp = labled_data.loc[labled_data['rumor'] == rumor].index.tolist()
        train_temp = labled_data.loc[labled_data['rumor'] != rumor].index.tolist()
        test_indices.append(test_temp)
        train_indices.append(train_temp)
    result = zip(train_indices,test_indices)
    return result

def grid_search_classifiers(labeled_data,validation,features_to_use=None,outfile=None):
    classifiers_types = ['knn','dtree','random_forest','max_ent','nb']
    scores = pd.DataFrame()
    print features_to_use
    for classifier in classifiers_types:
        if validation == 'kfold':
            s = k_fold_validation(labeled_data,classifier,features_to_use,verbose=False)
        elif validation == 'rumor':
            s = rumor_validation(labeled_data,classifier,features_to_use,verbose=False)
        s['classifier'] = classifier
        scores = scores.append(s)
    if outfile:
        scores.to_csv(outfile,encoding='utf-8')
    return scores

def learning_curve(df,classifier,n_ticks=10):
    target = df['class'].values.reshape(len(X),1)
    train_sizes= np.linspace(0, 1.0, n_ticks)
    classifier = Classifier.train_classifier(classifier_type)
    num_training_examples, train_scores, valid_scores = learning_curve(classifier,df,target,train_sizes=train_sizes)
    print valid_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='create uncertainty classifier')
    parser.add_argument('-i','--infile')
    parser.add_argument('-o','--outfile')
    parser.add_argument('-c','--classifier_type')
    parser.add_argument('-f','--features',nargs='*')
    parser.add_argument('-v','--validation')
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--grid_search',action='store_true')
    parser.add_argument('--learning_curve',action='store_true')
    args = parser.parse_args()
    df = pd.read_csv(args.infile,dtype={'text': object})
    df = df.loc[df['text'].notnull()]
    if not args.features:
        args.features = []
    if args.grid_search:
        grid_search_classifiers(df,args.validation,args.features,args.outfile)
    if args.learning_curve:
        learning_curve(df,args.classifier_type)
