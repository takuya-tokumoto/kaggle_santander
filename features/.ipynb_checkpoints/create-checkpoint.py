#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re as re
import sys
from base import Feature, get_arguments, generate_features
Feature.dir = 'features'

sampling_cnts = 1000000


def date_to_int(str_date):
    # 日付を数字に変換する関数です。 2015-01-28は 1, 2016-06-28は 18に変換します。
    
    Y, M, D = [int(a) for a in str_date.strip().split("-")] 
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date


'''
    読み込み・準備

'''

train = pd.read_feather('./data/input/train_ver2.feather')

test = pd.read_feather('./data/input/test_ver2.feather')

# 製品の変数を別途に保存しておきます。
prods = train.columns[24:].tolist()

# 24個の製品を1つも保有していない顧客のデータを除去します。
no_product = train[prods].sum(axis=1) == 0
train= train[~no_product]
## あとで消す
# メモリに乗らないのでサンプリング
train = train.sample(sampling_cnts)
print('学習データのサンプリング数： ',train.shape)

# 製品変数の欠損値をあらかじめ0に代替しておきます。
train[prods] = train[prods].fillna(0.0).astype(np.int8)

# 訓練データとテストデータを統合します。テストデータにない製品変数は0で埋めます。
for col in train.columns[24:]:
    test[col] = 0

'''
    trainとtestをunionしてデータフレームを作成

'''    
    
df = pd.concat([train, test], axis=0)


'''
    データ加工

'''

# カテゴリ変数を .factorize() 関数に通して label encodingします。
categorical_cols = ['ind_empleado', 'pais_residencia', 'sexo', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'nomprov', 'segmento']
for col in categorical_cols:
    df[col], _ = df[col].factorize(na_sentinel=-99)

# (特徴量エンジニアリング) 2つの日付変数から年度と月の情報を抽出します。
df['fecha_alta'].fillna(0.0, inplace=True)
df['fecha_alta_month'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['fecha_alta_year'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
df = df.drop(['fecha_alta'], axis =1)

df['ult_fec_cli_1t'].fillna(0.0, inplace=True)
df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
df['ult_fec_cli_1t'] = df['ult_fec_cli_1t'].astype(object)
df = df.drop(['ult_fec_cli_1t'], axis =1)

# それ以外の変数の欠損値をすべて -99に代替します。
df.fillna(-99, inplace=True)

# (特徴量エンジニアリング) lag-1 データを生成します。
# コード 2-12と類似したコードの流れです
# 日付を数字に変換し int_dateに保存します。
df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

# generate_features(globals(), args.force)

'''
    保存

'''

df = df.reset_index(drop=True)
df.to_feather('./data/input/' + 'df' + '.feather')