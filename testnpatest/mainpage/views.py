# -*- coding: utf-8 -*-
from django.shortcuts import render
import time
from django.http import *
# import nlp_model as nlp_m

# Create your views here.
m = 0
a = 0
def index(request):
    # nlp_m.analyze_doc()
    return render(request, 'mainpage/index.html', {'a': a})

def tags(request):
    return render(request, 'mainpage/tags.html', {'a': 'asdfadsfasdfasdfsadf'})