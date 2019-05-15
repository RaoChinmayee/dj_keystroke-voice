from django.contrib import admin
from django.urls import path
from . import views

app_name='mainpage'
urlpatterns = [
	path('',views.mainpage_view,name="mainpage"),
	path('update/',views.update,name="update"),
	path('voice/',views.voice,name="voice"),
	path('recorder/',views.recorder,name="recorder"),
	path('thankyou/',views.thankyou_view,name="thankyou"),
	path('Login_voice/',views.login_voice,name="Login_voice"),	
	path('recorder_temp/',views.recorder_temp,name="recorder_temp"),
	path('result/',views.result,name="result"),
	path('featuresExtraction_temp/',views.featuresExtraction_temp,name="featuresExtraction_temp"),
	path('end/',views.end,name="end"),



	]