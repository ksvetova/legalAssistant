# -*- coding: utf-8 -*-
from django.shortcuts import render
import time
from django.http import *

from .nlp_model import utils
# Uncomment last line in utils.py to initialize model #

# Create your views here.
m = 0
a = 1
def index(request):

    return render(request, 'mainpage/index.html', {'a': a, 'range': range(10)})

def tags(request):
    return render(request, 'mainpage/tags.html', {'a': 'asdfadsfasdfasdfsadf'})