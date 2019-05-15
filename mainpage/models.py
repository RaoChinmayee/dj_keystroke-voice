from django.db import models
from django.contrib.auth.models import User


class KeystrokeDB(models.Model):
    user_id = models.IntegerField(default=-1)
    user_name = models.CharField(max_length=30, default=None)
    sentence_index=models.IntegerField(default=0)
    for i in range(190):
        exec("chr%d_up=models.IntegerField(default=0)" % i)
        exec("chr%d_dn=models.IntegerField(default=0)" % i)

class KeystrokeDB1(models.Model):
    user_id = models.IntegerField(default=-1)
    user_name = models.CharField(max_length=30, default=None)
    sentence_index=models.IntegerField(default=0)
    for i in range(190):
        exec("chr%d_up=models.IntegerField(default=0)" % i)
        exec("chr%d_dn=models.IntegerField(default=0)" % i)


class user_list(models.Model):
	user_name = models.CharField(max_length=30, default=None)
	index = models.IntegerField(default=0)


class ParametersDB(models.Model):
    user_id = models.IntegerField(default=-1)
    user_name = models.CharField(max_length=30, default=None)
    sentence_index=models.IntegerField(default=0)
    char_index = models.IntegerField(default=-1)
    char_1 = models.IntegerField(default=0)
    char_2 = models.IntegerField(default=0)
    DD=models.IntegerField(default=0)
    DU=models.IntegerField(default=0)
    UD=models.IntegerField(default=0)
    UU=models.IntegerField(default=0)
    H1= models.IntegerField(default=0)
    H2= models.IntegerField(default=0)

class ParametersDB1(models.Model):
    user_id = models.IntegerField(default=-1)
    user_name = models.CharField(max_length=30, default=None)
    sentence_index=models.IntegerField(default=0)
    char_index = models.IntegerField(default=-1)
    char_1 = models.IntegerField(default=0)
    char_2 = models.IntegerField(default=0)
    DD=models.IntegerField(default=0)
    DU=models.IntegerField(default=0)
    UD=models.IntegerField(default=0)
    UU=models.IntegerField(default=0)
    H1= models.IntegerField(default=0)
    H2= models.IntegerField(default=0)

class voiceFeatures(models.Model):
    user_id = models.IntegerField(default=-1)
    user_name = models.CharField(max_length=30, default=None)
    frame_index = models.IntegerField(default=-1)
    for i in range(66):
        exec("f%d=models.FloatField(default=0)" % i)
 
class voiceFeatures_temp(models.Model):
    user_id = models.IntegerField(default=-1)
    user_name = models.CharField(max_length=30, default=None)
    frame_index = models.IntegerField(default=-1)
    for i in range(66):
        exec("f%d=models.FloatField(default=0)" % i)
