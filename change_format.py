import re
import json
import requests
import numpy as np
import pandas as pd


aspects = [
   "FOOD#PRICES",
   "FOOD#QUALITY",
   "FOOD#STYLE&OPTIONS",
   "DRINKS#PRICES",
   "DRINKS#QUALITY",
   "DRINKS#STYLE&OPTIONS",
   "RESTAURANT#PRICES",
   "RESTAURANT#GENERAL",
   "RESTAURANT#MISCELLANEOUS",
   "SERVICE#GENERAL",
   "AMBIENCE#GENERAL",
   "LOCATION#GENERAL"
]

def label_encoder(label):
    y = [np.nan] * len(aspects)
    ap_stm = re.findall('{(.+?), ([a-z]+)}', label)

    for aspect, sentiment in ap_stm:
        idx = aspects.index(aspect)
        y[idx] = sentiment

    return y

def txt2df(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as txt:
        data = txt.read().split('\n')

    df = pd.DataFrame()
    df['review'] = [review for review in data[1::4]]
    df[aspects] = [label_encoder(label) for label in data[2::4]]

    return df

def label_decoder(encoded_label):
    aps_stms = encoded_label[encoded_label.notna()]
    
    return ', '.join([f'{{{aspect}, {sentiment}}}' 
                      for aspect, sentiment in 
                      zip(aps_stms.index, aps_stms)])

def csv2str(filepath):
    df = pd.read_csv(filepath)
    rows = []
    for id, row in df.iterrows():
        review = row[0]
        labels = label_decoder(row[1:])
        rows.extend((f'#{id+1}', review, labels, ''))
    return '\n'.join(rows)


"""
    >>> root_dir = Path('CS221.M11.KHCL-Aspect-Based-Sentiment-Analysis/data')

    >>> train_txt_fp = root_dir/'original/1-VLSP2018-SA-Restaurant-train (7-3-2018).txt'
    >>> dev_txt_fp = root_dir/'original/2-VLSP2018-SA-Restaurant-dev (7-3-2018).txt'
    >>> test_txt_fp = root_dir/'original/3-VLSP2018-SA-Restaurant-test (8-3-2018).txt'

    >>> train_csv_fp = root_dir/'csv/train.csv'
    >>> dev_csv_fp = root_dir/'csv/dev.csv'
    >>> test_csv_fp = root_dir/'csv/test.csv'

    >>> assert train_txt_fp.is_file()
    >>> assert dev_txt_fp.is_file()
    >>> assert test_txt_fp.is_file()

    >>> train_df = txt2df(train_fp)
    >>> dev_df = txt2df(dev_fp)
    >>> test_df = txt2df(test_fp)

    >>> train_df.to_csv(train_csv_fp, index=False)
    >>> dev_df.to_csv(dev_csv_fp, index=False)
    >>> test_df.to_csv(test_csv_fp, index=False)
    
    >>> print(csv2str(train_csv_fp))
    >>> print(csv2str(dev_csv_fp))
    >>> print(csv2str(test_csv_fp))
"""
