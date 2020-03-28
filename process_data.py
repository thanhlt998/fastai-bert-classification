import pandas as pd
import os
from sklearn.model_selection import train_test_split
import string
import numpy as np
from sklearn.utils import shuffle


def list2text(texts):
    texts = list(texts)
    res = []
    for text in texts:
        text = text.strip()
        if not text: continue
        if text[-1] in string.punctuation:
            res.append(text)
        else:
            res.append(f'{text}.')
    return ' '.join(res)


def process(dir_name, output_dir, cols: list, test_pct=0.2, sep='\t'):
    train_dfs, test_dfs = [], []
    for fn in os.listdir(dir_name):
        df = pd.read_csv(os.path.join(dir_name, fn), sep=sep)
        label = fn.split('.')[0]
        sub_df = df[cols]
        sub_df.fillna(value='', inplace=True)
        sub_df['text'] = sub_df.apply(lambda x: list2text(x), axis=1)
        sub_df = sub_df[['text']]
        sub_df['label'] = label
        sub_df['text'].replace('', np.nan, inplace=True)
        sub_df.dropna(inplace=True)

        train_df, test_df = train_test_split(sub_df[['label', 'text']], test_size=test_pct)
        train_dfs.append(train_df)
        test_dfs.append(test_df)

    train_df = shuffle(pd.concat(train_dfs, ignore_index=True))
    test_df = shuffle(pd.concat(test_dfs, ignore_index=True))

    train_df.to_csv(os.path.join(output_dir, 'train.csv'), sep=sep, index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), sep=sep, index=False)


if __name__ == '__main__':
    input_dir = 'medical_data/raw_data'
    output_dir = 'medical_data/title'
    cols = ['title']
    process(dir_name=input_dir, output_dir=output_dir, cols=cols)


