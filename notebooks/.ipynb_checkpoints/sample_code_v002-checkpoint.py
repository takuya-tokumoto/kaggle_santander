#!/usr/bin/env python
# coding: utf-8

# # scripts/convert_to_feather.py

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
import gc

np.random.seed(2018)


# In[2]:


target = [
    'train_ver2',
    'test_ver2',
]

extension = 'csv'


# In[3]:


for t in target:
    
    print(f'=== {t} の作成開始')
    
    df = pd.read_csv('../data/input/' + t + '.' + extension, encoding="utf-8")
    
    ## データクレンジングしてからfeather化 ##
    # 不正な値があるカラム 'age', 'antiguedad', 'indrel_1mes', 'conyuemp'
    
    # 数値型変数の特異値と欠損値を -99に代替し、整数型に変換します。
    df['age'].replace(' NA', -99, inplace=True)
    df['age'] = df['age'].astype(np.int8)
    df['antiguedad'].replace('     NA', -99, inplace=True)
    df['antiguedad'] = df['antiguedad'].astype(np.int8)
    df['renta'].replace('         NA', -99, inplace=True)
    df['renta'].fillna(-99, inplace=True)
    df['renta'] = df['renta'].astype(float).astype(np.int8
    df['indrel_1mes'].replace('P', 5, inplace=True)
    df['indrel_1mes'].fillna(-99, inplace=True)
    df['indrel_1mes'] = df['indrel_1mes'].astype(float).astype(np.int8)
    df['conyuemp'].fillna('N', inplace=True) # Nが多いのでNで穴埋め
    
    df.to_feather('../data/input/' + t + '.feather')
    
    print(f'=== {t} の作成完了')


# # features/create.py

# In[1]:


import pandas as pd
import numpy as np
import re as re
import sys

sys.path.append('../features/')
from base import Feature, get_arguments, generate_features

Feature.dir = 'features'


# In[2]:


# 日付を数字に変換する関数です。 2015-01-28は 1, 2016-06-28は 18に変換します。
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")] 
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date


# In[12]:


# args = get_arguments()

'''
    読み込み・準備

'''

train = pd.read_feather('../data/input/train_ver2.feather')

test = pd.read_feather('../data/input/test_ver2.feather')

# 製品の変数を別途に保存しておきます。
prods = train.columns[24:].tolist()

# 24個の製品を1つも保有していない顧客のデータを除去します。
no_product = train[prods].sum(axis=1) == 0
train= train[~no_product]
## あとで消す
# メモリに乗らないのでサンプリング
train = train.sample(1000000)
print(train.shape)

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
df.to_feather('../data/input/' + 'df' + '.feather')


# # run.py

# In[1]:


import pandas as pd
import numpy as np
import re as re
import argparse
import json
import gc
import xgboost as xgb


# In[2]:


# import numpy as np

def apk(actual, predicted, k=7, default=0.0):
    # AP@7なので、最大7個まで使用します。
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        # 点数を付与する条件は次のとおり :
        # 予測値が正答に存在し (‘p in actual’)
        # 予測値に重複がなければ (‘p not in predicted[:i]’) 
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    # 正答値が空白である場合、ともかく 0.0点を返します。
    if not actual:
        return default

    # 正答の個数(len(actual))として average precisionを求めます。
    return score / min(len(actual), k)

def mapk(actual, predicted, k=7, default=0.0):
    # list of listである正答値(actual)と予測値(predicted)から顧客別 Average Precisionを求め, np.mean()を通して平均を計算します。
    return np.mean([apk(a, p, k, default) for a, p in zip(actual, predicted)]) 


# In[3]:


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='../configs/default.json')
# options = parser.parse_args()
options = parser.parse_args([])
config = json.load(open(options.config))

prods = config['target']
features = config['features']
params = config['model_params']


# In[4]:


df = pd.read_feather('../data/input/' + 'df' + '.feather')

# データをコピーし, int_date 日付に1を加え lagを生成します。変数名に _prevを追加します。
df_lag = df.copy()
df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns ]
df_lag['int_date'] += 1

# 原本データと lag データを ncodperと int_date を基準として合わせます。lag データの int_dateは 1 だけ押されているため、前の月の製品情報が挿入されます。
df_trn = df.merge(df_lag, on=['ncodpers','int_date'], how='left')

# メモリの効率化のために、不必要な変数をメモリから除去します。
del df, df_lag
gc.collect()


# In[5]:


# 前の月の製品情報が存在しない場合に備えて、0に代替します。
for prod in prods:
    prev = prod + '_prev'
    df_trn[prev].fillna(0, inplace=True)
df_trn.fillna(-99, inplace=True)

# lag-1 変数を追加します。
features += [feature + '_prev' for feature in features]
features += [prod + '_prev' for prod in prods]


# In[6]:


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


# In[7]:


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


# In[8]:


import logging
from lightgbm.callback import _format_eval_result


def log_best(model, metric):
    logging.debug(model.best_iteration)
    logging.debug(model.best_score['valid_0'][metric])


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list \
                and (env.iteration + 1) % period == 0:
            result = '\t'.join([
                _format_eval_result(x, show_stdv)
                for x in env.evaluation_result_list
            ])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback


# In[9]:


import lightgbm as lgb
import logging
import pickle

# from logs.logger import log_evaluation


def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params):

    # データセットを生成する
    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_eval  = xgb.DMatrix(X_valid, y_valid)
    
    watch_list = [(xgb_train, 'train'), (xgb_eval, 'eval')]

    logging.debug(params)

    # ロガーの作成
    logger = logging.getLogger('main')
    callbacks = [log_evaluation(logger, period=30)]

    
    # 上記のパラメータでモデルを学習する
    model = xgb.train(
        params, xgb_train,
        # モデルの評価用データを渡す
        evals=watch_list,
        # 最大で 1000 ラウンドまで学習する
        num_boost_round=10,
        # 10 ラウンド経過しても性能が向上しないときは学習を打ち切る
        early_stopping_rounds=20,
        # ログ
        # callbacks=callbacks
    )

    xgb_test = xgb.DMatrix(X_test)
    
    # テストデータを予測する
    y_pred = model.predict(xgb_test, ntree_limit=model.best_ntree_limit)
    
    # pickle.dump(model, open("xgb.baseline.pkl", "wb"))
    # best_ntree_limit = model.best_ntree_limit

    return y_pred, model


# In[11]:


# 訓練、検証データを XGBoost 形態に変換します。
X_trtrain = XY_trn[features].values
Y_trtrain = XY_trn['y'].values

X_trvalid = XY_vld[features].values
Y_trvalid = XY_vld['y'].values


# In[12]:


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


# In[13]:


# 検証データに対する予測値を求めます。
X_trtest = vld[features].values

preds_vld, _ = train_and_predict(X_trtrain, X_trvalid, Y_trtrain, Y_trvalid, X_trtest, params)

# 前の月に保有していた商品は新規購買が不可能なので、確率値からあらかじめ1を引いておきます。
preds_vld = preds_vld - vld[[prod + '_prev' for prod in prods]].values


# In[14]:


# 検証データの予測上位7個を抽出します。
result_vld = []
for ncodper, pred in zip(ncodpers_vld, preds_vld):
    y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods, range(len(prods)))]
    y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
    result_vld.append([ip for y,p,ip in y_prods])


# In[15]:


# 検証データから得ることのできる MAP@7 の最高点をあらかじめ求めておきます。(0.042663)
print('検証データから得ることのできる MAP@7 の最高点 : ', mapk(add_vld_list, add_vld_list, 7, 0.0))


# In[16]:


# 検証データの MAP@7の点数を求めます。(0.036466)
print('検証データの MAP@7の点数 : ',mapk(add_vld_list, result_vld, 7, 0.0))


# In[17]:


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
    num_boost_round=10,
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



# In[19]:


def make_submission(C, Y_test):

    """
    予測結果から提出用のcsvファイルを作成。
    
    
    Parameters
    --------
    C : list
        予測対象の顧客IDリスト(ncodpers)。
        
    Y_test : array
        各商品の獲得予測値。
    
    """
    submit_file = pd.DataFrame()
    C_list = []
    test_preds = []
    for ncodper, pred in zip(C, Y_test):
        y_prods = [(y,p,ip) for y,p,ip in zip(pred, prods, range(len(prods)))]
        y_prods = sorted(y_prods, key=lambda a: a[0], reverse=True)[:7]
        y_prods = [p for y,p,ip in y_prods]
        C_list.append(ncodper)
        test_preds.append(' '.join(y_prods))

    submit_file['ncodpers'] = C_list
    submit_file['added_products'] = test_preds
    submit_file.to_csv('collab_sub.csv', index=False)


# In[20]:


ncodpers_tst = tst['ncodpers'].values
preds_tst = preds_tst - tst[[prod + '_prev' for prod in prods]].values

make_submission(ncodpers_tst, preds_tst)


# In[ ]:




