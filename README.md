Santander Product Recommendation
===
- [サンタンデールコンペ](https://www.kaggle.com/c/santander-product-recommendation)についての解法を記載。
- ベースラインモデルと8位解法を模したスクリプトを格納。
- ただし計算負荷を下げるため学習データをサンプリングする処理を追加しているので注意。
- [kaggleコンペティション チャレンジブック](https://www.amazon.co.jp/s?k=kaggle%E3%82%B3%E3%83%B3%E3%83%9A%E3%83%86%E3%82%A3%E3%82%B7%E3%83%A7%E3%83%B3+%E3%83%81%E3%83%A3%E3%83%AC%E3%83%B3%E3%82%B8%E3%83%96%E3%83%83%E3%82%AF&adgrpid=113931757096&gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6VI2k-bgghlX2VNQpgcxxQAbAxN_vbeskmMmDtnUjTnJvr3gBJxa0aAnIiEALw_wcB&hvadid=649709208398&hvdev=c&hvlocphy=1009310&hvnetw=g&hvqmt=e&hvrand=6145891022325214495&hvtargid=kwd-1101980650705&hydadcr=1823_13591222&jp-ad-ap=0&tag=googhydr-22&ref=pd_sl_1ni4e43k14_e)を参照。


# Structures
```
.
├── configs
│   └── default.json
├── data
│   ├── input
│   │   ├── sample_submission.csv
│   │   ├── train_ver2.csv
│   │   └── test_ver2.csv
│   └── output
├── features
│   ├── __init__.py
│   ├── base.py
│   └── create.py
├── logs
│   └── logger.py
├── models
│   └── engines.py
│   └── xgboost.py
├── notebooks
│   └── eda.ipynb
├── scripts
│   └── clean.py
│   └── convert_to_feather.py
│   └── create_table.py
├── utils
│   └── __init__.py
│   └── eval_func.py
│   └── utils.py
├── .gitignore
├── .pylintrc
├── LICENSE
├── README.md
├── run.py
├── run_8th.py
└── tox.ini
```

# base-line moldel Commands
- ベースラインモデルの作成するための手順
## Change data to feather format

```
python scripts/convert_to_feather.py
```

## Create features

```
python features/create.py
```

## Run LightGBM

```
python run.py
```

# 8th moldel Commands
- 8位入賞モデルを作成するための手順
## Clension data

```
python scripts/clean.py
```

## Run

```
python run_8th.py
```


- './data/output' 配下の作成物をzip化しkaggleサイトのLate Submissionから提出  