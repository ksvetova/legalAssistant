# -*- coding: utf-8 -*-
from django.shortcuts import render
import time
from django.http import *

import mainpage.nlp_model as model

# Create your views here.
m = 0
a = 1
def index(request):
    model.test_function()
    return render(request, 'mainpage/index.html', {'a': a})

def tags(request):
    return render(request, 'mainpage/tags.html', {'a': 'asdfadsfasdfasdfsadf'})