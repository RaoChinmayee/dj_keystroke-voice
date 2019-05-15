
from django.contrib import admin
from django.urls import path, include
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('mainpage/',include('mainpage.urls'), name="mainpage"),
    path('Signup/',views.signup_view, name="Signup"),
    path('Authentication/',views.Authentication_view, name="Authentication"),
    path('Authentication1/',views.Authentication_view1, name="Authentication1"),
    path('Login/',views.login_view,name="Login"),
    
    path('Log/',views.log_view,name="Log"),
    path('homepage/',views.homepage,name="homepage"),
    path('logout/',views.logout_view,name="logout"), 
    path('update1/',views.update1,name="update1"),
    path('',views.homepage,name="homepage"),
]

urlpatterns += staticfiles_urlpatterns()