import csv
import io

from django.http import HttpResponse
from django.views.generic import FormView

from .forms import UploadForm

import pandas as pd
import numpy as np
import pickle


# Create your views here.
class UploadView(FormView):
    form_class = UploadForm
    template_name = 'app/UploadForm.html'

    def form_valid(self, form):
        #with open('test_x.csv',encoding="utf-8_sig") as csvfile:と同じ感じ
        #csvfile = io.TextIOWrapper(form.cleaned_data['file'] ,encoding="utf-8_sig")

        # データの読み込み
        test_x = pd.read_csv(form.cleaned_data['file'])
        train_x = pd.read_csv("train_x.csv")
        train_y = pd.read_csv("train_y.csv")

        # データの結合
        all_x = pd.concat([train_x, test_x], ignore_index=True, sort=False)
        all_df = pd.concat([all_x, train_y], axis=1)
        all_df = all_df.drop(columns=["お仕事No."])

        # EDA
        allnot_col = list(all_df.isnull().sum()[all_df.isnull().sum()==len(all_df)].index)
        all_df = all_df.drop(columns=allnot_col)

        one_col = list(all_df.nunique()[(all_df.nunique()==1)].index)
        onefull_col = all_df[one_col].isnull().sum()[all_df[one_col].isnull().sum()==0].index
        all_df = all_df.drop(columns=list(onefull_col))
        one_col2 = all_df.nunique()[(all_df.nunique()==1) ].index
        all_df[one_col2] = all_df[one_col2].fillna("NA")
        # 欠損値
        no_col_list = all_df.isnull().sum()[all_df.isnull().sum() > 0].index
        no_float_col = list(all_df[no_col_list].dtypes[all_df[no_col_list].dtypes=="float64"].index)
        no_float_col.remove("応募数 合計")
        no_obj_col = list(all_df[no_col_list].dtypes[all_df[no_col_list].dtypes=="object"].index)
        for col in no_float_col:
            all_df[col] = all_df[col].fillna(all_df[col].mean())
        for col in no_obj_col:
            all_df[col] = all_df[col].fillna(all_df[col].mode()[0])
        # LabelEncoding
        from sklearn.preprocessing import LabelEncoder
        cat_cols = list(all_df.dtypes[all_df.dtypes == "object"].index)
        for col in cat_cols:
            le = LabelEncoder()
            all_df[col] = le.fit_transform(all_df[col])

        # モデルの読み込み
        with open("model.pickle", mode="rb") as f:
            model = pickle.load(f)
        # 予測
        test = all_df[all_df["応募数 合計"].isnull()].drop(columns=["応募数 合計"])
        pred = model.predict(test)

        submission = pd.DataFrame({"お仕事No.": test_x["お仕事No."], "応募数 合計": pred})

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
