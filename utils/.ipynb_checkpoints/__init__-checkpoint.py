import pandas as pd

def make_submission(C, Y_test,prods):

    """
    予測結果から提出用のcsvファイルを作成。
    
    
    Parameters
    --------
    C : list
        予測対象の顧客IDリスト(ncodpers)。
        
    Y_test : array
        各商品の獲得予測値。
        
    prods : list
        ターゲットになる商品名。
    
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
    submit_file.to_csv('./data/output/20230303_collab_sub.csv', index=False)

