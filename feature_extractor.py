# coding: utf-8

import argparse
import numpy as np
import pandas as pd
import random
import os
import re

RAW_TWEETS_PATH = '/Users/klogg/research_data/ml_uncertainty/raw_tweets_all_rumors.csv'
SEED = 100

# import all data from mongo into a dataframe with columns text, class, and
# rumor
# pos = 1, neg = 0
def import_training_data(infile=None,balanced=True):
    count = 0
    column_types = {
        'text': object,
    }
    if infile:
        raw_tweets = pd.read_csv(infile,dtype=column_types)
    else:
        raw_tweets = pd.read_csv(RAW_TWEETS_PATH,dtype=column_types)
    raw_tweets = raw_tweets.loc[raw_tweets['text'].notnull()]
    raw_tweets = extract_features(raw_tweets)
    raw_tweets = clean_text(raw_tweets)
    if balanced:
        pos_examples = raw_tweets.loc[raw_tweets['class'] == 1]
        neg_examples = raw_tweets.loc[raw_tweets['class'] == 0].sample(len(pos_examples),random_state=SEED)
        examples = pos_examples.append(neg_examples)
    else:
        examples = raw_tweets
    examples = examples.reindex(np.random.permutation(examples.index))
    return examples
    
def extract_features(df):
    df = extract_question(df)
    df = extract_mention(df)
    df = extract_url(df)
    df = extract_hashtag(df)
    return df

def extract_question(df):
    df.loc[df['text'].str.contains('\?'), 'is_question'] = 1
    df.loc[~df['text'].str.contains('\?'), 'is_question'] = 0
    return df

def extract_mention(df):
    df.loc[df['text'].str.contains(r'RT .*?:'), 'has_mention'] = 1
    df.loc[df['text'].str.contains(r'"@.*?:'), 'has_mention'] = 1
    df.loc[df['text'].str.contains(ur'\u201c' + '@.*?:'), 'has_mention'] = 1
    df.loc[df['text'].str.contains(r'via @.*?:'), 'has_mention'] = 1
    df.loc[df['text'].str.contains(r'via @.*?'), 'has_mention'] = 1
    df.loc[df['text'].str.contains(r'@.*?'), 'has_mention'] = 1
    df.loc[df['has_mention'] != 1, 'has_mention'] = 0
    return df

def extract_url(df):
    df.loc[df['text'].str.contains('http.*?\s|http.*?$'), 'has_url'] = 1
    df.loc[~df['text'].str.contains('http.*?\s|http.*?$'), 'has_url'] = 0
    return df

def extract_hashtag(df):
    df.loc[df['text'].str.contains('#'), 'has_hashtag'] = 1
    df.loc[~df['text'].str.contains('#'), 'has_hashtag'] = 0
    return df

def clean_text(df):
    while True:
        temp_df = df.copy()
        df['text'] = df['text'].apply(lambda text: clean_url(text))
        df['text'] = df['text'].apply(lambda text: clean_mention(text))
        if df['text'].equals(temp_df['text']):
            return df

def clean_mention(text):
    s = ur'\u201c' + '@.*?:'
    sub = r'RT .*?:|"@.*?:|via @.*?:|via @.*?\s|@.*?\s|@.*?$|' + s
    text = re.sub(sub,'',text).strip()
    return text

def clean_url(text):
    sub = '@.*?\s|http.*?$'
    text = re.sub(sub,'',text).strip()
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='process dataset for uncertainty')
    parser.add_argument('-i','--infile')
    parser.add_argument('-o','--outfile')
    parser.add_argument('--balanced',action='store_true')
    args = parser.parse_args()
    df = import_training_data(args.infile,args.balanced)
    df.to_csv(args.outfile,encoding='utf-8',index=False)
