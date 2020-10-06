"""website_RD URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    #path('admin/', admin.site.urls),
    #path('admin/', views.admin),
    path('datastatistics/', views.admin),
    #path(r'^$', views.home),
    path('', views.home),
    path('home/', views.home),
    path('ChineseRD/', views.ChineseRD),
    path('EnglishRD/', views.EnglishRD),
    path('feedback/', views.feedback),
    path('about/', views.about),
    path('about_en/', views.about_en),
    path('papers/', views.papers),
    path('help/', views.help),
    path('GetChDefis/', views.GetChDefis),
    path('GetEnDefis/', views.GetEnDefis),
    path('ChineseRDCluster/', views.ChineseRDCluster),
    path('EnglishRDCluster/', views.EnglishRDCluster),
    #path('/', views.),
]
