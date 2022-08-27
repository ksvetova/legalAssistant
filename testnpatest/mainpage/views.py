# -*- coding: utf-8 -*-
from django.shortcuts import render
import time
from django.http import *

# Create your views here.
m = 0
a = 1
def index(request):
    return render(request, 'mainpage/index.html', {'a': a})

def tags(request):
    return render(request, 'mainpage/tags.html', {'a': 'asdfadsfasdfasdfsadf'})