import csv
import io

from django.http import HttpResponse
from django.views.generic import FormView
from .forms import UploadForm

import pandas as pd 

from django.shortcuts import render
from .module import pred
from django.shortcuts import redirect

from .models import Post


class UploadView(FormView):
    form_class = UploadForm
    template_name = 'app/UploadForm.html'

    def form_valid(self, form):
        #データの読み込み
        test_x = pd.read_csv(form.cleaned_data['file'], na_values=["なし"])
        #予測をファイルに保存
        submit_values = pred(test_x)
        submit_values.to_csv("submit.csv", index=False, encoding="utf-8")

        return redirect("move")

         
# Download.htmlに移動
def move(request):
    return render(request, "app/Download.html", {})

# 結果をcsvファイルでダウンロード
def export(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename = "result.csv"'

    # 保存したファイルから結果をダウンロード
    submit = pd.read_csv("submit.csv", encoding="utf-8")
    submit_values = submit.values.tolist()
    submit_columns = submit.columns.tolist()
    submit_values.insert(0, submit_columns)

    writer = csv.writer(response)  
    # for row in submit_values:
    #     writer.writerow(row)

    for post in Post.objects.all():
        writer.writerow([post.number, post.value])

    return response