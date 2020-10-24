import pandas as pd 
import numpy as np
import re 
import pickle
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
import MeCab
from sklearn.preprocessing import LabelEncoder

def pred(test_x):    
    #データの読み込み
    test_x = test_x
    train_x = pd.read_pickle("pickles/train_x.pickle")
    train_y = pd.read_pickle("pickles/train_y.pickle")

    all_x = pd.concat([train_x, test_x], ignore_index=True, sort=False)
    all_df = pd.concat([all_x, train_y], axis=1)

    #前処理
    all_df = preprocess(all_df)

    test = all_df[all_df["応募数mean"].isnull()].drop(columns=["応募数mean", "お仕事No."])

    #予測
    with open("pickles/stacked_model.pickle", mode="rb") as ff:
        model = pickle.load(ff)
    pred = model.predict(test)
    
    # 予測値を学習データにある値に変換
    def getNearestValue(list, num):
        idx = np.abs(np.asarray(list) - num).argmin()
        return list[idx]
    values = list(train_y["応募数 合計"].unique())
    pred = np.array([getNearestValue(values, a) for a in pred])

    submission = pd.DataFrame({"お仕事No.": test_x["お仕事No."], "応募数 合計": pred})

    # 予測値をリストで返す
    submit_values = submission.values.tolist()
    submit_columns = submission.columns.tolist()
    submit_values.insert(0, submit_columns)
    
    return submit_values




def preprocess(all_df):
    # すべて欠損しているカラムは削除
    allnot_col = list(all_df.isnull().sum()[all_df.isnull().sum()==19244].index)
    all_df = all_df.drop(columns=allnot_col)

    # 値がすべて同じカラムかつ欠損値を持たないカラムは削除
    one_col = list(all_df.nunique()[all_df.nunique()==1].index)
    onefull_col = list(all_df[one_col].isnull().sum()[all_df[one_col].isnull().sum()==0].index)
    all_df = all_df.drop(columns=onefull_col)

    # 重複データ
    train = all_df[all_df["応募数 合計"].notnull()]
    test = all_df[all_df["応募数 合計"].isnull()]
    train['応募数mean'] = train.groupby(["お仕事No."])["応募数 合計"].transform("mean")
    test["応募数mean"] = np.nan
    all_df = pd.concat([train, test], ignore_index=True, sort=False)
    # 全重複数もカラムとして残しておく
    all_df["all_count"] = all_df.groupby(["お仕事No."])["お仕事No."].transform("count")
    train = all_df[all_df["応募数mean"].notnull()]
    test = all_df[all_df["応募数mean"].isnull()]
    train = train.drop(columns=["応募数 合計"])
    test = test.drop(columns=["応募数 合計"])
    train = train.drop_duplicates(subset=["お仕事No."])
    all_df = pd.concat([train, test], ignore_index=True, sort=False)

    # 欠損している数を表すカラムを追加
    all_df["NaN_num"] = all_df.isnull().sum(axis=1)

    # 欠損かどうかを表すカラムを追加
    no_df = pd.DataFrame({"num": all_df.isnull().sum()[all_df.isnull().sum()>0].values,
    "type": all_df[all_df.isnull().sum()[all_df.isnull().sum()>0].index].dtypes},
    index=all_df.isnull().sum()[all_df.isnull().sum()>0].index)
    for i in no_df.index:
        all_df["NaN_"+i] = np.where(all_df[i].isnull(), 1, 0)
    all_df = all_df.drop(columns=["（派遣先）概要　事業内容","NaN_応募数mean"])

    # 男女比　女は削除
    all_df = all_df.drop(columns=["（派遣先）配属先部署　男女比　女", "NaN_（派遣先）配属先部署　男女比　女"])

    # 欠損値の補完
    no_df2 = pd.DataFrame({"num": all_df.isnull().sum()[all_df.isnull().sum()>0].values,
    "type": all_df[all_df.isnull().sum()[all_df.isnull().sum()>0].index].dtypes},
    index=all_df.isnull().sum()[all_df.isnull().sum()>0].index)
    # 欠損している数値カラム
    no_float_col = list(no_df2[no_df2["type"] != "object"].index)
    no_float_col.remove("応募数mean")
    # 欠損しているカテゴリカラム
    no_obj_col = list(no_df2[no_df2["type"] == "object"].index)
    # カテゴリ変数と考えられるものはNAで補完
    cols = ["（紹介予定）入社後の雇用形態", "勤務地　最寄駅2（駅からの交通手段）",
    "勤務地　最寄駅1（駅からの交通手段）"
    ]
    for col in cols:
        all_df[col] = all_df[col].fillna("NA")
    # 数値変数として扱いたいものは-9999で補完¶
    cols2 = ["（派遣先）配属先部署　男女比　男", "（派遣先）配属先部署　人数",
    "勤務地　最寄駅1（分）", "（派遣先）配属先部署　平均年齢",
    "給与/交通費　給与上限", "勤務地　最寄駅2（分）"
    ]
    for col in cols2:
        all_df[col] = all_df[col].fillna(-9999)
    # 欠損値をNAで埋める
    for col in no_obj_col[:-1]:
        all_df[col] = all_df[col].fillna("NA")

    # 数値変数をカテゴリ変数に
    for col in ["フラグオプション選択", "職種コード", "会社概要　業界コード", "仕事の仕方", "勤務地　市区町村コード"]:
        all_df[col] = all_df[col].astype(str)

    all_df['掲載期間　開始日'] = pd.to_datetime(all_df['掲載期間　開始日'], format="%Y/%m/%d")
    all_df['掲載期間　終了日'] = pd.to_datetime(all_df['掲載期間　終了日'], format="%Y/%m/%d")
    all_df['期間・時間　勤務開始日'] = pd.to_datetime(all_df['期間・時間　勤務開始日'], format="%Y/%m/%d")

    # 掲載開始日と勤務開始日
    all_df['掲載期間　開始日'] = pd.to_datetime(all_df['掲載期間　開始日'], format="%Y/%m/%d")
    all_df['掲載期間　終了日'] = pd.to_datetime(all_df['掲載期間　終了日'], format="%Y/%m/%d")
    all_df['期間・時間　勤務開始日'] = pd.to_datetime(all_df['期間・時間　勤務開始日'], format="%Y/%m/%d")
    all_df["勤務開始-掲載開始"] = (all_df['期間・時間　勤務開始日'] - all_df['掲載期間　開始日'])
    all_df["勤務開始-掲載開始"] = all_df["勤務開始-掲載開始"].dt.days
    all_df = all_df.drop(columns=['掲載期間　開始日', "掲載期間　終了日", "期間・時間　勤務開始日"])
    # 勤務時間
    all_df["workingstart"] = all_df["期間・時間　勤務時間"].str.split("〜", expand=True)[0]
    all_df["workingend"] = all_df["期間・時間　勤務時間"].str.split("〜", expand=True)[1].str.split("　", expand=True)[0]
    all_df["workingstart"] = pd.to_datetime(all_df['workingstart'], format='%H:%M')
    all_df["workingend"] = pd.to_datetime(all_df['workingend'], format='%H:%M')
    all_df["workingtime_m"] = (all_df["workingend"] - all_df["workingstart"]).astype('timedelta64[m]')

    all_df["workingrest"] = all_df["期間・時間　勤務時間"].str.split("休憩", expand=True)[1].str.split("分", expand=True)[0].str.split("<BR>", expand=True)[0]
    all_df["workingrest"] = all_df["workingrest"].apply(lambda x: re.sub(r'\D', '', str(x)))
    all_df["workingrest"][all_df["workingrest"]==""] = "０"
    all_df["workingrest"][all_df["workingrest"]=="１"] = "６０"
    all_df["workingrest"][all_df["workingrest"]=="６０９０"] = "７５"
    all_df["workingrest"][all_df["workingrest"]=="１３３０１６００"] = "０"
    all_df["workingrest"][all_df["workingrest"]=="１０３０１３３０"] = "０"
    all_df["workingrest"] = all_df["workingrest"].apply(int)
    all_df["productiontime_m"] = (all_df["workingtime_m"] - all_df["workingrest"])#.astype('timedelta64[m]')
    for i in list(all_df.dtypes[all_df.dtypes=="datetime64[ns]"].index):
        all_df[i] = all_df[i].astype(str)

    # NaN_期間･時間　備考を追加したので削除
    all_df = all_df.drop(columns=["期間･時間　備考"])

    #言語処理
    token_df = pd.read_pickle("pickles/train_token.pickle")
    #お仕事名
    char_filters = [UnicodeNormalizeCharFilter(),
                    RegexReplaceCharFilter(r"[!$%&\'()*+,-./:;<=>?@\\^_`{|}~◆▼★②●☆■★【】『』「」、♪≪≫]", " ")
                ]
    token_filters = [CompoundNounFilter(),
                    POSKeepFilter(['名詞']),
                    LowerCaseFilter()
                    ]
    a = Analyzer(char_filters=char_filters,  token_filters=token_filters)
    length = all_df["応募数mean"].notnull().sum()
    all_df['お仕事名_token'] = np.nan
    all_df['お仕事名_token'][:length] = token_df["お仕事名_token"]
    all_df['お仕事名_token'][length:] = all_df["お仕事名"][length:].apply(lambda x: " ".join([token.surface for token in a.analyze(x)]))
    #all_df['お仕事名_token'] = token_df["お仕事名_token"]

    with open("pickles/grid_1.pickle", mode="rb") as ff:
        model = pickle.load(ff)
    all_df["お仕事名_pred"] = model.predict(all_df['お仕事名_token'].values)
    all_df = all_df.drop(columns=['お仕事名_token'])

    #仕事内容
    select_conditions = ['名詞']
    tagger = MeCab.Tagger('')
    tagger.parse('')
    def wakati_text(text):
        node = tagger.parseToNode(text)
        terms = []
        while node:
            term = node.surface
            pos = node.feature.split(',')[0]
            if pos in select_conditions:
                terms.append(term)
            node = node.next
        text_result = ' '.join(terms)
        return text_result

    length = all_df["応募数mean"].notnull().sum()
    all_df['仕事内容_token']  = np.nan
    all_df['仕事内容_token'][:length] = token_df["仕事内容_token"]
    all_df['仕事内容_token'][length:] = all_df["仕事内容"][length:].apply(wakati_text)
    #all_df['仕事内容_token'] = token_df["仕事内容_token"]

    with open("pickles/grid_2.pickle", mode="rb") as ff:
        model = pickle.load(ff)
    all_df["仕事内容_pred"] = model.predict(all_df['仕事内容_token'])
    all_df = all_df.drop(columns=['仕事内容_token'])

    #お仕事のポイント（仕事PR）
    char_filters = [UnicodeNormalizeCharFilter(),
                    RegexReplaceCharFilter(r"[!$%&\'()*+,-./:;<=>?@\\^_`{|}~◆▼★②●☆■★【】『』「」、♪≪≫]", " ")
                ]
    token_filters = [CompoundNounFilter(),
                    POSKeepFilter(['名詞']),
                    LowerCaseFilter()
                    ]
    a = Analyzer(char_filters=char_filters,  token_filters=token_filters)
    length = all_df["応募数mean"].notnull().sum()
    print(length)
    all_df['お仕事のポイント_token'] = np.nan
    all_df['お仕事のポイント_token'][:length] = token_df["お仕事のポイント_token"]
    all_df['お仕事のポイント_token'][length:] = all_df["お仕事のポイント（仕事PR）"][length:].apply(lambda x: " ".join([token.surface for token in a.analyze(x)]))
    #all_df['お仕事のポイント_token'] = token_df["お仕事のポイント_token"]

    with open("pickles/grid_3.pickle", mode="rb") as ff:
        model = pickle.load(ff)
    all_df["お仕事のポイント_pred"] = model.predict(all_df['お仕事のポイント_token'].values)
    all_df = all_df.drop(columns=['お仕事のポイント_token'])

    # （派遣先）配属先部署
    char_filters = [UnicodeNormalizeCharFilter(),
                    RegexReplaceCharFilter(r"[!$%&\'()*+,-./:;<=>?@\\^_`{|}~◆▼★②●☆■★【】『』「」、♪≪≫]", " ")
                ]
    token_filters = [POSKeepFilter(['名詞']),
                    LowerCaseFilter()
                    ]
    a = Analyzer(char_filters=char_filters,  token_filters=token_filters)
    length = all_df["応募数mean"].notnull().sum()
    all_df['（派遣先）配属先部署_token'] = np.nan
    all_df['（派遣先）配属先部署_token'][:length] = token_df["（派遣先）配属先部署_token"]
    all_df['（派遣先）配属先部署_token'][length:] = all_df["（派遣先）配属先部署"][length:].apply(lambda x: " ".join([token.surface for token in a.analyze(x)]))
    #all_df['（派遣先）配属先部署_token'] = token_df["（派遣先）配属先部署_token"]

    with open("pickles/grid_4.pickle", mode="rb") as ff:
        model = pickle.load(ff)
    all_df["（派遣先）配属先部署_pred"] = model.predict(all_df['（派遣先）配属先部署_token'])
    all_df = all_df.drop(columns=['（派遣先）配属先部署_token'])

    # （派遣先）職場の雰囲気
    char_filters = [UnicodeNormalizeCharFilter(),
                    RegexReplaceCharFilter(r"[!$%&\'()*+,-./:;<=>?@\\^_`{|}~◆▼★②●☆■★【】『』「」、♪≪≫]", " ")
                ]
    token_filters = [CompoundNounFilter(),
                    POSKeepFilter(['名詞']),
                    LowerCaseFilter()
                    ]
    a = Analyzer(char_filters=char_filters,  token_filters=token_filters)
    length = all_df["応募数mean"].notnull().sum()
    all_df['（派遣先）職場の雰囲気_token'] = np.nan
    all_df['（派遣先）職場の雰囲気_token'][:length] = token_df["（派遣先）職場の雰囲気_token"]
    all_df['（派遣先）職場の雰囲気_token'][length:] = all_df["（派遣先）職場の雰囲気"][length:].apply(lambda x: " ".join([token.surface for token in a.analyze(x)]))
    #all_df['（派遣先）職場の雰囲気_token'] = token_df["（派遣先）職場の雰囲気_token"]

    with open("pickles/grid_5.pickle", mode="rb") as ff:
        model = pickle.load(ff)
    all_df["（派遣先）職場の雰囲気_pred"] = model.predict(all_df['（派遣先）職場の雰囲気_token'])

    with open("pickles/lda_5.pickle", mode="rb") as ff:
        lda = pickle.load(ff)
    X_lda = lda.transform(all_df["（派遣先）職場の雰囲気_token"])
    all_df["（派遣先）職場の雰囲気_lda"] = X_lda.argmax(axis=1)
    all_df = all_df.drop(columns=['（派遣先）職場の雰囲気_token'])

    all_df = all_df.drop(columns=["お仕事名",  "仕事内容", "お仕事のポイント（仕事PR）", "（派遣先）配属先部署", "（派遣先）職場の雰囲気"])

    # Label Encoding
    cat_cols = list(all_df.dtypes[all_df.dtypes == "object"].index)
    for col in cat_cols:
        le = LabelEncoder()
        all_df[col] = le.fit_transform(all_df[col].apply(lambda x: str(x)))

    return all_df


        # # 予測
        # with open("xgb_model.pickle", mode="rb") as ff:
        #     xgb_model = pickle.load(ff)
        # with open("lgb_model.pickle", mode="rb") as ff:
        #     lgb_model = pickle.load(ff)
        # pred_an = xgb_model.predict(test)*0.5 + lgb_model.predict(test)*0.5
        # pred_df = pd.DataFrame({"お仕事No.": test_x["お仕事No."], "応募数 合計": pred_an})

        # # リーク
        # leak_df = all_df_labeled.copy() 
        # leak_df['応募数 合計'] = leak_df.groupby(["お仕事No."])["応募数mean"].transform(np.nanmean)
        # test_df = leak_df.iloc[len(train):, :]

        # leak_test = test_df[test_df["応募数 合計"].notnull()].drop(columns=["お仕事No.","応募数mean"])
        # noleak_test = test_df[test_df["応募数 合計"].isnull()].drop(columns=["お仕事No.","応募数mean","応募数 合計"])

        # noleak_test["応募数 合計"] = xgb_model.predict(noleak_test)*0.5 + lgb_model.predict(noleak_test)*0.5

        # leak_df = pd.concat([leak_test, noleak_test], sort=True) # インデックスで並び替え

        # # リークと予測値の平均をとる
        # pred = leak_df["応募数 合計"].values*0.5 + pred_df["応募数 合計"].values*0.5
        # pred = np.where(pred<0, 0.0, pred)
        # submission = pd.DataFrame({"お仕事No.": test_x["お仕事No."].values, "応募数 合計": pred})

    
