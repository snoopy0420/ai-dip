import csv
import io

from django.http import HttpResponse
from django.views.generic import FormView

from .forms import UploadForm

import pandas as pd 
import numpy as np
import re
from janome.tokenizer import Tokenizer
from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Lasso
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import pickle


# Create your views here.
class UploadView(FormView):
    form_class = UploadForm
    template_name = 'app/UploadForm.html'

    def form_valid(self, form):
        #データの読み込み
        train_x = pd.read_csv("train_x.csv", na_values=["なし"])
        train_y = pd.read_csv("train_y.csv" )
        test_x = pd.read_csv(form.cleaned_data['file'], na_values=["なし"])
        all_x = pd.concat([train_x, test_x], ignore_index=True, sort=False)
        train_y = train_y.drop(columns=["お仕事No."])
        all_df = pd.concat([all_x, train_y], axis=1)

        # 全て欠損しているカラム
        allnot_col = list(all_df.isnull().sum()[all_df.isnull().sum()==19244].index)
        all_df = all_df.drop(columns=allnot_col)
        # 値がすべて同じカラム
        one_col = list(all_df.nunique()[all_df.nunique()==1].index)
        onefull_col = list(all_df[one_col].isnull().sum()[all_df[one_col].isnull().sum()==0].index)
        all_df = all_df.drop(columns=onefull_col)
        # 重複データ
        train = all_df[all_df["応募数 合計"].notnull()]
        test = all_df[all_df["応募数 合計"].isnull()]
        train['応募数mean'] = train.groupby(["お仕事No."])["応募数 合計"].transform("mean")
        test["応募数mean"] = np.nan
        all_df = pd.concat([train, test], ignore_index=True, sort=False)
        all_df["all_count"] = all_df.groupby(["お仕事No."])["お仕事No."].transform("count")
        train = all_df[all_df["応募数mean"].notnull()]
        test = all_df[all_df["応募数mean"].isnull()]
        train = train.drop(columns=["応募数 合計"])
        test = test.drop(columns=["応募数 合計"])
        train = train.drop_duplicates(subset=["お仕事No."])
        all_df = pd.concat([train, test], ignore_index=True, sort=False)

        # 行ごとの欠損値の数
        all_df["NaN_num"] = all_df.isnull().sum(axis=1)

        # カラムごとに欠損かどうかを表す2値変数
        no_df = pd.DataFrame({"num": all_df.isnull().sum()[all_df.isnull().sum()>0].values,
                            "type": all_df[all_df.isnull().sum()[all_df.isnull().sum()>0].index].dtypes},
                            index=all_df.isnull().sum()[all_df.isnull().sum()>0].index)
        for i in no_df.index:
            all_df["NaN_"+i] = np.where(all_df[i].isnull(), 1, 0)
        all_df = all_df.drop(columns=["（派遣先）概要　事業内容","NaN_応募数mean"])

        # 男女比　女は削除
        all_df = all_df.drop(columns=["（派遣先）配属先部署　男女比　女", "NaN_（派遣先）配属先部署　男女比　女"])

        # 数値変数をカテゴリ変数に
        for col in ["フラグオプション選択", "職種コード", "会社概要　業界コード", "仕事の仕方", "勤務地　市区町村コード"]:
            all_df[col] = all_df[col].astype(str)

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

        token_df = pd.read_csv("train_token.csv")

        # お仕事名
        char_filters = [UnicodeNormalizeCharFilter(),
                        RegexReplaceCharFilter(r"[!$%&\'()*+,-./:;<=>?@\\^_`{|}~◆▼★②●☆■★【】『』「」、♪≪≫]", " ")
                    ]
        token_filters = [CompoundNounFilter(),
                        POSKeepFilter(['名詞']),
                        LowerCaseFilter()
                        ]
        a = Analyzer(char_filters=char_filters,  token_filters=token_filters)
        length = all_df["応募数mean"].notnull().sum()
        all_df['お仕事名_token'] = all_df['お仕事名']
        all_df['お仕事名_token'][:length] = token_df["お仕事名_token"]
        all_df['お仕事名_token'][length:] = all_df["お仕事名"][length:].apply(lambda x: " ".join([token.surface for token in a.analyze(x)]))

        with open("grid_1.pickle", mode="rb") as ff:
            model = pickle.load(ff)
        all_df["お仕事名_pred"] = model.predict(all_df['お仕事名_token'])
        all_df = all_df.drop(columns=['お仕事名_token'])

        # 仕事内容
        char_filters = [UnicodeNormalizeCharFilter(),
                        RegexReplaceCharFilter(r"[!$%&\'()*+,-./:;<=>?@\\^_`{|}~◆▼★②●☆■★【】『』「」、♪≪≫]", " ")
                    ]
        token_filters = [CompoundNounFilter(),
                        POSKeepFilter(['名詞']),
                        LowerCaseFilter()
                        ]
        a = Analyzer(char_filters=char_filters,  token_filters=token_filters)
        length = all_df["応募数mean"].notnull().sum()
        all_df['仕事内容_token']  = np.nan
        all_df['仕事内容_token'][:length] = token_df["仕事内容_token"]
        all_df['仕事内容_token'][length:] = all_df["仕事内容"][length:].apply(lambda x: " ".join([token.surface for token in a.analyze(x)]))

        with open("grid_2.pickle", mode="rb") as ff:
            model = pickle.load(ff)
        all_df["仕事内容_pred"] = model.predict(all_df['仕事内容_token'])
        all_df = all_df.drop(columns=['仕事内容_token'])

        # お仕事のポイント（仕事PR）
        char_filters = [UnicodeNormalizeCharFilter(),
                        RegexReplaceCharFilter(r"[!$%&\'()*+,-./:;<=>?@\\^_`{|}~◆▼★②●☆■★【】『』「」、♪≪≫]", " ")
                    ]
        token_filters = [CompoundNounFilter(),
                        POSKeepFilter(['名詞']),
                        LowerCaseFilter()
                        ]
        a = Analyzer(char_filters=char_filters,  token_filters=token_filters)
        length = all_df["応募数mean"].notnull().sum()
        all_df['お仕事のポイント_token'] = np.nan
        all_df['お仕事のポイント_token'][:length] = token_df["お仕事のポイント_token"]
        all_df['お仕事のポイント_token'][length:] = all_df["お仕事のポイント（仕事PR）"][length:].apply(lambda x: " ".join([token.surface for token in a.analyze(x)]))

        with open("grid_3.pickle", mode="rb") as ff:
            model = pickle.load(ff)
        all_df["お仕事のポイント_pred"] = model.predict(all_df['お仕事のポイント_token'])
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

        with open("grid_4.pickle", mode="rb") as ff:
            model = pickle.load(ff)
        all_df["（派遣先）配属先部署_pred"] = model.predict(all_df['（派遣先）配属先部署_token'])
        all_df = all_df.drop(columns=['（派遣先）配属先部署_token'])
        all_df = all_df.drop(columns=["お仕事名",  "仕事内容", "お仕事のポイント（仕事PR）", "（派遣先）配属先部署"])

        # Label Encoding
        cat_cols = list(all_df.dtypes[all_df.dtypes == "object"].index)
        all_df_labeled = all_df.copy()

        for col in cat_cols:
            le = LabelEncoder()
            all_df_labeled[col] = le.fit_transform(all_df_labeled[col].apply(lambda x: str(x)))

        test = all_df_labeled[all_df_labeled["応募数mean"].isnull()].drop(columns=["お仕事No.","応募数mean"])

        # 予測
        with open("xgb_model.pickle", mode="rb") as ff:
            xgb_model = pickle.load(ff)
        with open("lgb_model.pickle", mode="rb") as ff:
            lgb_model = pickle.load(ff)
        pred_an = xgb_model.predict(test)*0.5 + lgb_model.predict(test)*0.5
        pred_df = pd.DataFrame({"お仕事No.": test_x["お仕事No."], "応募数 合計": pred_an})

        # リーク
        leak_df = all_df_labeled.copy() 
        leak_df['応募数 合計'] = leak_df.groupby(["お仕事No."])["応募数mean"].transform(np.nanmean)
        test_df = leak_df.iloc[len(train):, :]

        leak_test = test_df[test_df["応募数 合計"].notnull()].drop(columns=["お仕事No.","応募数mean"])
        noleak_test = test_df[test_df["応募数 合計"].isnull()].drop(columns=["お仕事No.","応募数mean","応募数 合計"])

        noleak_test["応募数 合計"] = xgb_model.predict(noleak_test)*0.5 + lgb_model.predict(noleak_test)*0.5

        leak_df = pd.concat([leak_test, noleak_test], sort=True) # インデックスで並び替え

        # リークと予測値の平均をとる
        pred = leak_df["応募数 合計"].values*0.5 + pred_df["応募数 合計"].values*0.5
        pred = np.where(pred<0, 0.0, pred)
        submission = pd.DataFrame({"お仕事No.": test_x["お仕事No."].values, "応募数 合計": pred})


        # submissionをリストに変換
        submit_values = submission.values.tolist()
        submit_columns = submission.columns.tolist()
        submit_values.insert(0, submit_columns)

        # 結果をcsvファイルでダウンロード
        response = HttpResponse(content_type='csv')
        response['Content-Disposition'] = 'attachment; filename = "result.csv"'
        
        writer = csv.writer(response)  
        for row in submit_values:
            writer.writerow(row)

        return response
