#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import xgboost as xgb
import gc

np.random.seed(2018)

target = [
    'train_ver2',
    'test_ver2',
]

extension = 'csv'

for t in target:
    
    print(f'=== {t} の作成開始')

    df = pd.read_csv('./data/input/' + t + '.' + extension, encoding="utf-8")

    ## データクレンジングしてからfeather化 ##
    # 不正な値があるカラム 'age', 'antiguedad', 'indrel_1mes', 'conyuemp'

    # 数値型変数の特異値と欠損値を -99に代替し、整数型に変換します。
    df['age'].replace(' NA', -99, inplace=True)
    df['age'] = df['age'].astype(np.int8)
    df['antiguedad'].replace('     NA', -99, inplace=True)
    df['antiguedad'] = df['antiguedad'].astype(np.int8)
    df['renta'].replace('         NA', -99, inplace=True)
    df['renta'].fillna(-99, inplace=True)
    df['renta'] = df['renta'].astype(float).astype(np.int8)
    df['indrel_1mes'].replace('P', 5, inplace=True)
    df['indrel_1mes'].fillna(-99, inplace=True)
    df['indrel_1mes'] = df['indrel_1mes'].astype(float).astype(np.int8)
    df['conyuemp'].fillna('N', inplace=True) # Nが多いのでNで穴埋め

    df.to_feather('./data/input/' + t + '.feather')

    print(f'=== {t} の作成完了')