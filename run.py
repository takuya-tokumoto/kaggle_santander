import pandas as pd
import datetime
import logging
from sklearn.model_selection import KFold
import argparse
import json
import numpy as np

from utils import load_datasets, load_target
from logs.logger import log_best
from models.lgbm import train_and_predict


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='./configs/default.json')
options = parser.parse_args()
config = json.load(open(options.config))

now = datetime.datetime.now()
logging.basicConfig(
    filename='./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now), level=logging.DEBUG
)
logging.debug('./logs/log_{0:%Y%m%d%H%M%S}.log'.format(now))

feats = config['features']
logging.debug(feats)

target_name = config['target_name']

X_train_all, X_test = load_datasets(feats)
y_train_all = load_target(target_name)
logging.debug(X_train_all.shape)

y_preds = []
models = []

lgbm_params = config['lgbm_params']


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
# X_trn = XY_trn.as_matrix(columns=features)
# Y_trn = XY_trn.as_matrix(columns=['y'])
X_trn = XY_trn[features].values
Y_trn = XY_trn['y'].values
dtrn = xgb.DMatrix(X_trn, label=Y_trn, feature_names=features)

# X_vld = XY_vld.as_matrix(columns=features)
# Y_vld = XY_vld.as_matrix(columns=['y'])
X_vld = XY_vld[features].values
Y_vld = XY_vld['y'].values
dvld = xgb.DMatrix(X_vld, label=Y_vld, feature_names=features)

# # kf = KFold(n_splits=3, random_state=0)
# kf = KFold(n_splits=3, random_state=0, shuffle = True)
# for train_index, valid_index in kf.split(X_train_all):
#     X_train, X_valid = (
#         X_train_all.iloc[train_index, :], X_train_all.iloc[valid_index, :]
#     )
#     y_train, y_valid = y_train_all[train_index], y_train_all[valid_index]

#     # lgbmの実行
#     y_pred, model = train_and_predict(
#         X_train, X_valid, y_train, y_valid, X_test, lgbm_params
#     )

#     # 結果の保存
#     y_preds.append(y_pred)
#     models.append(model)

#     # スコア
#     log_best(model, config['loss'])

# CVスコア
scores = [
    m.best_score['valid_0'][config['loss']] for m in models
]
score = sum(scores) / len(scores)
print('===CV scores===')
print(scores)
print(score)
logging.debug('===CV scores===')
logging.debug(scores)
logging.debug(score)

# submitファイルの作成
ID_name = config['ID_name']
sub = pd.DataFrame(pd.read_csv('./data/input/test.csv')[ID_name])

y_sub = sum(y_preds) / len(y_preds)

if y_sub.shape[1] > 1:
    y_sub = np.argmax(y_sub, axis=1)

sub[target_name] = y_sub

sub.to_csv(
    './data/output/sub_{0:%Y%m%d%H%M%S}_{1}.csv'.format(now, score),
    index=False
)
