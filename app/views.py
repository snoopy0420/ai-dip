import csv
import io

from django.views.generic import FormView
from .forms import UploadForm

import pandas as pd 
from .module import pred

from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import redirect

from .models import Post


class UploadView(FormView):
    form_class = UploadForm
    template_name = 'app/UploadForm.html'

    def form_valid(self, form):
        #アップロードされたデータの読み込み
        test_x = pd.read_csv(form.cleaned_data['file'], na_values=["なし"])
        #予測
        submit_values = pred(test_x)

        #データベース内のオブジェクトを削除
        p = Post.objects.all()
        p.delete()
        #データベースに保存
        for line in submit_values[1:]:
            Post.objects.create(number=line[0], value=line[1])

        return redirect("move")

         
# Download.htmlに移動
def move(request):
    return render(request, "app/Download.html", {})

# 結果をcsvファイルでダウンロード
def export(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename = "result.csv"'
    writer = csv.writer(response)  

    #カラムを入れる
    writer.writerow(["お仕事No.", "応募数　合計"])
    #データベースから取り出す
    for post in Post.objects.all():
        writer.writerow([post.number, post.value])

    return response