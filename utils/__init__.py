import pandas as pd


def load_datasets(feats):
    dfs = [pd.read_feather(f'features/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1, sort=False)
    dfs = [pd.read_feather(f'features/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1, sort=False)
    return X_train, X_test


def load_target(target_name):
    train = pd.read_csv('./data/input/train.csv')
    y_train = train[target_name]
    
    '''
        新規購買件数を目的変数に追加
        
    '''
    
    # データをコピーし, int_date 日付に1を加え lagを生成します。変数名に _prevを追加します。
    df_lag = df.copy()
    df_lag.columns = [col + '_prev' if col not in ['ncodpers', 'int_date'] else col for col in df.columns ]
    df_lag['int_date'] += 
    
    return y_train
