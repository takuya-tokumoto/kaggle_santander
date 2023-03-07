import pandas as pd
import numpy as np
import gc


np.random.seed(2018)


# データを呼び出します。
trn = pd.read_csv('./data/input/train_ver2.csv')
tst = pd.read_csv('./data/input/test_ver2.csv')


## データの前処理 ##

# 製品の変数を別途に保存しておきます。
prods = trn.columns[24:].tolist()

# 製品変数の欠損値をあらかじめ0に代替しておきます。
trn[prods] = trn[prods].fillna(0.0).astype(np.int8)

# 24個の製品を1つも保有していない顧客のデータを除去します。
no_product = trn[prods].sum(axis=1) == 0
trn = trn[~no_product]

# 訓練データとテストデータを統合します。テストデータにない製品変数は0で埋めます。
for col in trn.columns[24:]:
    tst[col] = 0
df = pd.concat([trn, tst], axis=0)

# 学習に使用する変数を入れるlistです。
features = []

# カテゴリ変数を .factorize() 関数に通して label encodingします。
categorical_cols = ['ind_empleado', 'pais_residencia', 'sexo', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp', 'canal_entrada', 'indfall', 'tipodom', 'nomprov', 'segmento']
for col in categorical_cols:
    df[col], _ = df[col].factorize(na_sentinel=-99)
features += categorical_cols

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

# 学習に使用する数値型変数を featuresに追加します。
features += ['age','antiguedad','renta','ind_nuevo','indrel','indrel_1mes','ind_actividad_cliente']

# (特徴量エンジニアリング) 2つの日付変数から年度と月の情報を抽出します。
df['fecha_alta_month'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['fecha_alta_year'] = df['fecha_alta'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
features += ['fecha_alta_month', 'fecha_alta_year']

df['ult_fec_cli_1t_month'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[1])).astype(np.int8)
df['ult_fec_cli_1t_year'] = df['ult_fec_cli_1t'].map(lambda x: 0.0 if x.__class__ is float else float(x.split('-')[0])).astype(np.int16)
features += ['ult_fec_cli_1t_month', 'ult_fec_cli_1t_year']

# それ以外の変数の欠損値をすべて -99に代替します。
df.fillna(-99, inplace=True)

# (特徴量エンジニアリング) lag-1 データを生成します。
# コード 2-12と類似したコードの流れです

# 日付を数字に変換する関数です。 2015-01-28は 1, 2016-06-28は 18に変換します。
def date_to_int(str_date):
    Y, M, D = [int(a) for a in str_date.strip().split("-")] 
    int_date = (int(Y) - 2015) * 12 + int(M)
    return int_date

# 日付を数字に変換し int_dateに保存します。
df['int_date'] = df['fecha_dato'].map(date_to_int).astype(np.int8)

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

# # lag-1 変数を追加します。
# features += [feature + '_prev' for feature in features]
# features += [prod + '_prev' for prod in prods]

df.to_feather('./data/input/' + 'df' + '.feather')
df_trn.to_feather('./data/input/' + 'df_trn' + '.feather')
df_lag.to_feather('./data/input/' + 'df_lag' + '.feather')
