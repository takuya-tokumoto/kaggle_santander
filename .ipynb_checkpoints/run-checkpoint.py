#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re as re
import argparse
import json
import gc
import logging
import xgboost as xgb
import sys
from utils.eval_func import *
from models.xgboost import train_and_predict
from utils.__init__ import *

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args([])
config = json.load(open(options.config))

prods = config['target']
features = config['features']
params = config['model_params']

df = pd.read_feather('./data/input/' + 'df' + '.feather')

# データをコピーし, int_date 日付に1を加え lagを生成します。変数名に _prevを追加します。
df_lag = df.copy()
df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns ]
df_lag['int_date'] += 1

# 原本データと lag データを ncodperと int_date を基準として合わせます。lag データの int_dateは 1 だけ押されているため、前の月の製品情報が挿入されます。
df_trn = df.merge(df_lag, on=['ncodpers','int_date'], how='left')

# メモリの効率化のために、不必要な変数をメモリから除去します。
del df, df_lag
gc.collect()

# 前の月の製品情報が存在しない場合に備えて、0に代替します。
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)
df_trn.fillna(-99, inplace=True)

# lag-1 変数を追加します。
features += [feature + '_prev' for feature in features]
features += [prod + '_prev' for prod in prods]

###
### Baseline モデル以後、多様な特徴量エンジニアリングをここに追加します。
###

## モデル学習
# 学習のため、データを訓練、検証用に分離します。
# 学習には 2016-01-28 ~ 2016-04-28 のデータだけを使用し、検証には 2016-05-28 のデータを使用します。
use_dates = ['2016-01-28', '2016-02-28', '2016-03-28', '2016-04-28', '2016-05-28']
trn = df_trn[df_trn['fecha_dato'].isin(use_dates)]
tst = df_trn[df_trn['fecha_dato'] == '2016-06-28']
del df_trn

# 訓練データから新規購買件数だけを抽出します。
X = []
Y = []
for i, prod in enumerate(prods):
    prev = prod + '_prev'
    prX = trn[(trn[prod] == 1) & (trn[prev] == 0)]
    prY = np.zeros(prX.shape[0], dtype=np.int8) + i
    X.append(prX)
    Y.append(prY)
XY = pd.concat(X)
Y = np.hstack(Y)
XY['y'] = Y

# 訓練、検証データに分離します。
vld_date = '2016-05-28'
XY_trn = XY[XY['fecha_dato'] != vld_date]
XY_vld = XY[XY['fecha_dato'] == vld_date]

# 訓練、検証データを XGBoost 形態に変換します。
X_trtrain = XY_trn[features].values
Y_trtrain = XY_trn['y'].values

X_trvalid = XY_vld[features].values
Y_trvalid = XY_vld['y'].values

# MAP@7 評価基準のための準備作業です。
# 顧客識別番号を抽出します。
vld = trn[trn['fecha_dato'] == vld_date]
ncodpers_vld = vld['ncodpers'].values
# 検証データから新規購買を求めます。
for prod in prods:
    prev = prod + '_prev'
    padd = prod + '_add'
    vld[padd] = vld[prod] - vld[prev]    
add_vld = vld[[prod + '_add' for prod in prods]].values
add_vld_list = [list() for i in range(len(ncodpers_vld))]

# 顧客別新規購買正答値を add_vld_listに保存し、総 countを count_vldに保存します。
count_vld = 0
for ncodper in range(len(ncodpers_vld)):
    for prod in range(len(prods)):
        if add_vld[ncodper, prod] > 0:
            add_vld_list[ncodper].append(prod)
            count_vld += 1


# 検証データに対する予測値を求めます。
X_trtest = vld[features].values

preds_vld, eval_model = train_and_predict(X_trtrain, X_trvalid, Y_trtrain, Y_trvalid, X_trtest, params)

# 前の月に保有していた商品は新規購買が不可能なので、確率値からあらかじめ1を引いておきます。
preds_vld = preds_vld - vld[[prod + '_prev' for prod in prods]].values

# 検証データの予測上位7個を抽出します。
result_vld = []
for ncodper, pred in zip(ncodpers_vld, preds_vld):
    y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods, range(len(prods)))]
    y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
    result_vld.append([ip for y,p,ip in y_prods])

# 検証データから得ることのできる MAP@7 の最高点をあらかじめ求めておきます。(0.042663)
print('検証データから得ることのできる MAP@7 の最高点 : ', mapk(add_vld_list, add_vld_list, 7, 0.0))

# 検証データの MAP@7の点数を求めます。(0.036466)
print('検証データの MAP@7の点数 : ',mapk(add_vld_list, result_vld, 7, 0.0))

# XGBoost モデルを全体の訓練データで学習します。
X_all = XY[features].values
Y_all = XY['y'].values
dall = xgb.DMatrix(X_all, label=Y_all, feature_names=features)
watch_list = [(dall, 'train')]

# # ツリーの個数を増加したデータの量に比例して増やします。
# best_ntree_limit = int(best_ntree_limit * (len(XY_trn) + len(XY_vld)) / len(XY_trn))
# XGBoost モデル再学習！
# model = xgb.train(params, dall, num_boost_round=best_ntree_limit, evals=watch_list)
model = xgb.train(
    params, 
    dall, 
    evals=watch_list,
    # 最大で 1000 ラウンドまで学習する
    num_boost_round=eval_model.best_ntree_limit,
    # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
    early_stopping_rounds=20,
)

# 変数の重要度を出力してみます。予想していた変数が上位に来ていますか？
print("Feature importance:")
for kv in sorted([(k,v) for k,v in model.get_fscore().items()], key=lambda kv: kv[1], reverse=True):
    print(kv)

# Kaggleに提出するため、テストデータに対する予測値を求めます。
X_tst = tst[features].values
dtst = xgb.DMatrix(X_tst, feature_names=features)
preds_tst = model.predict(dtst, ntree_limit=model.best_ntree_limit)

ncodpers_tst = tst['ncodpers'].values
preds_tst = preds_tst - tst[[prod + '_prev' for prod in prods]].values

make_submission(ncodpers_tst, preds_tst, prods)
