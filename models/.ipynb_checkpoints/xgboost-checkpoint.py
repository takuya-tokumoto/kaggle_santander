import argparse
import logging
import json
import gc
import xgboost as xgb

def train_and_predict(X_train, X_valid, y_train, y_valid, X_test, params):

    # データセットを生成する
    xgb_train = xgb.DMatrix(X_train, y_train)
    xgb_eval  = xgb.DMatrix(X_valid, y_valid)
    
    watch_list = [(xgb_train, 'train'), (xgb_eval, 'eval')]

    logging.debug(params)

    # ロガーの作成
    # logger = logging.getLogger('main')
    # callbacks = [log_evaluation(logger, period=30)]

    
    # 上記のパラメータでモデルを学習する
    model = xgb.train(
        params, xgb_train,
        # モデルの評価用データを渡す
        evals=watch_list,
        # 最大で 1000 ラウンドまで学習する
        num_boost_round=1000,
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