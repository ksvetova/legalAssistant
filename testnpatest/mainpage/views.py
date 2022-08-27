# -*- coding: utf-8 -*-
from django.shortcuts import render
from django.http import *

# Create your views here.

def index(request):
    return render(request, 'mainpage/index.html')

def tags(request):
    return render(request, 'mainpage/tags.html', {'title': 'О теге'})