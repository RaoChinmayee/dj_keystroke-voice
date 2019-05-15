from django.shortcuts import render,redirect
from .models import user_list,KeystrokeDB,ParametersDB,voiceFeatures,voiceFeatures_temp
from django.http import HttpResponse,JsonResponse
import time
import json
import re
import pandas as pd
import pyaudio
import wave
from array import array
from scipy.io import wavfile
import scipy.signal
from scipy.io.wavfile import write 
import numpy as np
import sqlite3
import librosa
import time
from datetime import timedelta as td
from python_speech_features import mfcc
from python_speech_features import logfbank,fbank,ssc
from pyAudioAnalysis import audioFeatureExtraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from channels import Channel

from pydub import AudioSegment


def fun(features,F_vectors,j):
	for i in range(65):
		exec("features.f%d=F_vectors[%d][%d]" % (i,j,i))
	features.save()

def featuresExtraction(username1):
	print("sucessssssssssssssss")
	rate, data= wavfile.read("Filtered_audio.wav")
	F_vectors=mfcc(data,rate,nfft=1103,numcep=13)
	f_vectors1=logfbank(data,rate,nfft=1103)
	f_vectors3=ssc(data,rate,nfft=1103)
	f_vectors1 = list(f_vectors1)
	f_vectors3 = list(f_vectors3)
	#F_vectors=np.transpose(F_vectors)
	F_vectors = np.array((F_vectors))
	length = F_vectors.shape[0]
	F_vectors= list(F_vectors)
	for i in range (length):
		F_vectors[i] = list(F_vectors[i])
		f_vectors1[i] = list(f_vectors1[i])
		F_vectors[i].extend(f_vectors1[i])
		f_vectors3[i] = list(f_vectors3[i])
		F_vectors[i].extend(f_vectors3[i])
		
	username=username1['username1']
	name=user_list.objects.get(user_name=username)
	userid=name.id
	print(time.time())
	for j in range(length):
		#print("j>>",j)
		features=voiceFeatures.objects.create(user_name=username,user_id=userid,frame_index =j)
		features=voiceFeatures.objects.get(user_name=username,user_id=userid,frame_index =j)
		#for i in range(65):
			#exec("features.f%d=F_vectors[%d][%d]" % (i,j,i))

		fun(features,F_vectors,j)
		
	print("done")
	print(time.time())










def keystroke_update(dictionary):
	i=0
	j=0
	ch_index=0
	time_list={}
	char_list=[]
	time_list=dictionary['time_list1']
	list_index=dictionary['list_index1']
	username=dictionary['username1']
	#print(time_list,'......................time_list')
	#print(list_index,'......................list_index')
	#print(username,'......................username')
	
	for d in range(5):
		#print(time_list[d])

		
		name=user_list.objects.get(user_name=str(username))
		userid=name.id
		sentence_index=name.index
		instance=KeystrokeDB.objects.get_or_create(user_name=str(username), user_id=userid, sentence_index=d)
		instance=KeystrokeDB.objects.get(user_name=str(username),user_id=userid, sentence_index=d)

		
		list_index1=int(list_index[d])
		key_list = list(time_list[d].keys())
		for i in range (0, list_index1):
			key_split1=re.split('[0-9]+',key_list[i])
			#print(key_split1)
			if(key_split1[2]=='dn'):
				if(key_split1[1]=='Backspace'):
					char_list.append(8)
				elif(key_split1[1]=='Shift'):
					char_list.append(16)
				elif(key_split1[1]=="CapsLock"):
					char_list.append(20)
				elif(key_split1[1]=="Tab"):
					char_list.append(9)
				elif(key_split1[1]=="Control"):
					char_list.append(17)
				elif(key_split1[1]=="Alt"):
					char_list.append(18)

				elif(key_split1[1]=='ArrowLeft'):
					char_list.append(37)
				elif(key_split1[1]=='ArrowRight'):
					char_list.append(39)
				elif(key_split1[1]=='Enter'):
					char_list.append(13)
				else:
					char_list.append(ord(key_split1[1]))

				#print('in down',key_split1[1])
				#print('in down if',key_split1[1])
				temp=time_list[d][key_list[i]]
				exec("instance.chr%d_dn=temp" % ch_index)
				for j in range(i,list_index1):
					key_split2=re.split('[0-9]+',key_list[j])
					if(key_split2[2]=='up'):
					#print('in up',key_split2[1])
						if(key_split1[1] == key_split2[1]):
							exec("instance.chr%d_up= time_list[d][key_list[j]]" % ch_index)
							ch_index=ch_index+1
							break
		res=1
		res=instance.save()
		print(res,'res..................................')

		i=0

		print(char_list)
		for i in range (0, ch_index-1):

			parameters=ParametersDB.objects.create(user_name=str(username),user_id=userid, char_index = i, sentence_index=d)
			parameters=ParametersDB.objects.get(user_name=str(username),user_id=userid, char_index = i, sentence_index=d)
			parameters.char_1=char_list[i]
			parameters.char_2= char_list[i+1]
			exec("parameters.DD= instance.chr%d_dn-instance.chr%d_dn" % (i+1,i))
			exec("parameters.UD= instance.chr%d_dn-instance.chr%d_up" % (i+1,i))
			exec("parameters.DU= instance.chr%d_up-instance.chr%d_dn" % (i+1,i))
			exec("parameters.UU= instance.chr%d_up-instance.chr%d_up" % (i+1,i))
			exec("parameters.H1= instance.chr%d_up-instance.chr%d_dn" % (i,i))
			exec("parameters.H2= instance.chr%d_up-instance.chr%d_dn" % (i+1,i+1))
			i=i+1
			parameters.save()
			#print('sentence index...........',sentence_index)
		char_list.clear()
		ch_index=0
	print('done')




def featuresExtraction_temp(username1):
	
	rate, data= wavfile.read("Filtered_audio.wav")
	F_vectors=mfcc(data,rate,nfft=1103,numcep=13)
	f_vectors1=logfbank(data,rate,nfft=1103)
	f_vectors3=ssc(data,rate,nfft=1103)
	f_vectors1 = list(f_vectors1)
	f_vectors3 = list(f_vectors3)
	F_vectors = np.array((F_vectors))
	length = F_vectors.shape[0]
	F_vectors= list(F_vectors)
	for i in range (length):
		F_vectors[i] = list(F_vectors[i])
		f_vectors1[i] = list(f_vectors1[i])
		F_vectors[i].extend(f_vectors1[i])
		f_vectors3[i] = list(f_vectors3[i])
		F_vectors[i].extend(f_vectors3[i])
	username=username1['username1']
	name=user_list.objects.get(user_name=str(username))
	userid=name.id
	voiceFeatures_temp.objects.filter(user_id=userid).delete()
	print(time.time())
	for j in range(length):
		#print("j>>",j)
		features=voiceFeatures_temp.objects.create(user_name=username,user_id=userid,frame_index =j)
		features=voiceFeatures_temp.objects.get(user_name=username,user_id=userid,frame_index =j)
		'''or i in range(65):
			exec("features.f%d=F_vectors[%d][%d]" % (i,j,i))
		features.save()'''
		fun(features,F_vectors,j)

	print("done")
	print(time.time())
	return redirect('Authentication')
	



