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
from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta,freqz

import random
from djangonautic_ks import my_settings

from pydub import AudioSegment
# Create your views here.


sentence=['qwerty']



def mainpage_view(request):

	
	if request.method=='POST':
		#validat

		

		return redirect('thankyou')
	else:
		#print("...................................in mainpage GET index",index)
		#convert_modeltocsv()
		
		return render(request, 'mainpage/mainpage.html',{'data':sentence[0]})
		
def voice(request):
	return render(request, 'voice.html')

def login_voice(request):
	return render(request, 'Login_voice.html')


def result(request):
	return render(request, 'result.html')


def thankyou_view(request):
	return render(request,'thankyou.html')





def update(request):
	if request.method=='POST':
		if 'dict' in request.POST:
			time_list= request.POST.get('dict')
			time_list= json.loads(time_list)
			list_index=request.POST.getlist('list_index[]')			
			#print(list_index[1])
			username=request.user.username
			Channel('keystroke_update').send({ "time_list1":time_list,"list_index1":list_index,"username1":username})
			return redirect('mainpage:voice')
		else:
			return HttpResponse('fail')


	
'''def update(request):
	i=0
	j=0
	ch_index=0
	time_list={}
	char_list=[]
	print(request.POST)
	if request.method=='POST':

		if 'dict' in request.POST:
			#print(request.POST)
			#print(time_list)
			time_list= request.POST.get('dict')
			time_list= json.loads(time_list)
			list_index=request.POST.getlist('list_index[]')			
			print(list_index[1])

			for d in range(5):
				print(time_list[d])

				username=request.user.username
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

				instance.save()
				#print(char_list)

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
		return redirect('mainpage:voice')
	else:
		return HttpResponse('fail')'''

def bandpass_firwin(ntaps, lowcut, highcut, fs, window='hamming'):
    nyq = 0.5 * fs
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=window, scale=False)
    return taps

def bandpass_kaiser(ntaps, lowcut, highcut, fs, width):
    nyq = 0.5 * fs
    atten = kaiser_atten(ntaps, width / nyq)
    beta = kaiser_beta(atten)
    taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                  window=('kaiser', beta), scale=False)
    return taps

def bandpass_remez(ntaps, lowcut, highcut, fs, width):
    delta = 0.5 * width
    edges = [0, lowcut - delta, lowcut + delta,
             highcut - delta, highcut + delta, 0.5 * fs]
    taps = remez(ntaps, edges, [0, 1, 0], Hz=fs)
    return taps


def recorder(request):

	THRESHOLD = 500
	CHUNK_SIZE = 1024
	FORMAT = pyaudio.paInt16
	RATE = 44100
	WAVE_OUTPUT_FILENAME = "file.wav"

	if request.method=='POST':
		audio = pyaudio.PyAudio()
		stream= audio.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
		frames=[]
		for i in range(0, int(RATE/CHUNK_SIZE*30)):
			data=stream.read(1024)
			data_chunk=array('h',data)
			vol=max(data_chunk)
			
			
			frames.append(data)
			
		stream.stop_stream()
		stream.close()
		audio.terminate() 
		waveFile= wave.open(WAVE_OUTPUT_FILENAME,'wb')
		waveFile.setnchannels(1)
		waveFile.setsampwidth(audio.get_sample_size(FORMAT))
		waveFile.setframerate(RATE)
		waveFile.writeframes(b''.join(frames))
		waveFile.close()

		rate, data= wavfile.read("file.wav")
		#audio_clip = data/32768
		n_rate, n_data= wavfile.read("file.wav")
		noise_clip = n_data/32768
		output = removeNoise(audio_clip=data,rate=rate)
		output = np.array(output)
		write("Filtered_audio.wav", rate, output)
		user_name1=request.user.username
		Channel('featuresExtraction').send(dict(
            username1=user_name1,
        ))
		return HttpResponse('success')

	else:
		return HttpResponse('fail')


def removeNoise(audio_clip,rate):
   
    xn= audio_clip
    fourier_xn = np.fft.rfft(xn)
    fs = rate *2
    lowcut=300
    highcut = 2000
    ntaps = 128
    taps_hamming = bandpass_firwin(ntaps, lowcut, highcut, fs=fs)
    taps_kaiser10 = bandpass_kaiser(ntaps, lowcut, highcut, fs=fs, width=1.0)
    remez_width = 1.0
    taps_remez = bandpass_remez(ntaps, lowcut, highcut, fs=fs, width=remez_width)
    w, h1 = freqz(taps_hamming, 1, worN=len(fourier_xn))
    w1, h2 = freqz(taps_kaiser10, 1, worN=len(fourier_xn))
    w2, h3 = freqz(taps_remez, 1, worN=len(fourier_xn))
    freq = fourier_xn
    filtered = freq*h1    #choosing the window, h1=> hamming, h2=> kaiser, h3=> remez
    filtered_time_domain = np.fft.irfft(filtered)
    return filtered_time_domain

    

def featuresExtraction(request):
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
		F_vectors[i].append(t_name)
	username=request.user.username
	name=user_list.objects.get(user_name=request.session['name'])
	userid=name.id
	for j in range(length):
		print("j>>",j)
		features=voiceFeatures.objects.create(user_name=request.session['name'],user_id=userid,frame_index =j)
		features=voiceFeatures.objects.get(user_name=request.session['name'],user_id=userid,frame_index =j)
		for i in range(65):
			exec("features.f%d=F_vectors[%d][%d]" % (i,j,i))
		features.save()


def recorder_temp(request):
	print(request.session['name'],'session............................')
	voiceFeatures.objects.filter(user_id=4).delete()
	ParametersDB.objects.filter(user_id=4).delete()
	KeystrokeDB.objects.filter(user_id=4).delete()
	THRESHOLD = 500
	CHUNK_SIZE = 1024
	FORMAT = pyaudio.paInt16
	RATE = 44100
	WAVE_OUTPUT_FILENAME = "file.wav"
	text_v=''
	text1='correct user'
	text2='incorrect user'

	if request.method=='POST':
		audio = pyaudio.PyAudio()
		stream= audio.open(format=FORMAT, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)
		frames=[]
		for i in range(0, int(RATE/CHUNK_SIZE*6)):
			data=stream.read(1024)
			data_chunk=array('h',data)
			vol=max(data_chunk)
			
			
			frames.append(data)
			
		stream.stop_stream()
		stream.close()
		audio.terminate() 
		waveFile= wave.open(WAVE_OUTPUT_FILENAME,'wb')
		waveFile.setnchannels(1)
		waveFile.setsampwidth(audio.get_sample_size(FORMAT))
		waveFile.setframerate(RATE)
		waveFile.writeframes(b''.join(frames))
		waveFile.close()

		rate, data= wavfile.read("file.wav")
		#audio_clip = data/32768
		n_rate, n_data= wavfile.read("file.wav")
		noise_clip = n_data/32768
		output = removeNoise(audio_clip=data, rate=rate)
		output = np.array(output)
		write("Filtered_audio.wav", rate, output)


		'''featuresExtraction_temp(request)
		username=request.user.username
		name=user_list.objects.get(user_name=request.session['name'])
		userid=name.id
'success'
		voiceFeatures_temp.objects.filter(user_id=userid).delete()
		
		text_v=Random_forest_v()

		if (text_v==text1):	
#		if (text==text1 and text_v==text1):
			print("in iffffffffffffffffff")
			response=0
			return HttpResponse(response)
		elif(text_v==text2):
			response=1'''
		return HttpResponse('success')
	else:
		return HttpResponse('fail')


def featuresExtraction_temp(request):
	
	user_name1=request.session['name']
	if request.method=='POST':

		print("im in ")
		ret=Channel('featuresExtraction_temp').send(dict(
	            username1=user_name1,
	        ))
		print(ret)
		
		return HttpResponse('success')

	else:
		return HttpResponse('fail')

	'''file='test_v.csv'
	table='mainpage_voiceFeatures_temp'
	convert_modeltocsv_voice(userid,userid,file,table)
	text_v=Random_forest_v()

	if (text_v==text1):	
#		if (text==text1 and text_v==text1):
		print("in iffffffffffffffffff")
		response=0
		return HttpResponse(response)
	elif(text_v==text2):
		response=1
		return HttpResponse(response)'''
	
	

def end(request):
	response=0
	text_v=''
	text1='correct user'
	text2='incorrect user'
	file='test_v.csv'
	table='mainpage_voiceFeatures_temp'
	name=user_list.objects.get(user_name=request.session['name'])
	userid=name.id
	convert_modeltocsv_voice(userid,userid,file,table)
	response1=Random_forest_v()
	response2=request.session['res1']
	print(response1)
	print(response2)
	resp=(response1+response2)/2
	print(resp)
	if(response1>60 and response2>65):
		response=1
		return HttpResponse(response)
	else:
		if(resp>=65):
			response=1
			return HttpResponse(response)

		else:
			response=0
			return HttpResponse(response)


	


def convert_modeltocsv_voice(id1,id2,file,table):
	with open(file,'w+') as write_file:
		conn = sqlite3.connect('db.sqlite3')
		att_row='user_id'+','+'user_name'+','+'frame_index'+','+'f0'+','+'f1'+','+'f2'+','+'f3'+','+'f4'+','+'f5'+','+'f6'+','+'f7'+','+'f8'+','+'f9'+','+'f10'+','+'f11'+','+'f12'+','+'f13'+','+'f14'+','+'f15'+','+'f16'+','+'f17'+','+'f18'+','+'f19'+','+'f20'+','+'f21'+','+'f22'+','+'f23'+','+'f24'+','+'f25'+','+'f26'+','+'f27'+','+'f28'+','+'f29'+','+'f30'+','+'f31'+','+'f32'+','+'f33'+','+'f34'+','+'f35'+','+'f36'+','+'f37'+','+'f38'+','+'f39'+','+'f40'+','+'f41'+','+'f42'+','+'f43'+','+'f44'+','+'f45'+','+'f46'+','+'f47'+','+'f48'+','+'f49'+','+'f50'+','+'f51'+','+'f52'+','+'f53'+','+'f54'+','+'f55'+','+'f56'+','+'f57'+','+'f58'+','+'f59'+','+'f60'+','+'f61'+','+'f62'+','+'f63'+','+'f64'+'\n'
		cursor=conn.cursor()

		write_file.write(att_row)

		
		for row in cursor.execute('SELECT * FROM {} WHERE (user_id <= ?) AND (user_id >= ?) '.format(table),(id2,id1)):
			temp_row=str(row[1])+','+str(row[2])+','+str(row[3])+','+str(row[4])+','+str(row[5])+','+str(row[6])+','+str(row[7])+','+str(row[8])+','+str(row[9])+','+str(row[10])+','+str(row[11])+','+str(row[12])+','+str(row[13])+','+str(row[14])+','+str(row[15])+','+str(row[16])+','+str(row[17])+','+str(row[18])+','+str(row[19])+','+str(row[20])+','+str(row[21])+','+str(row[22])+','+str(row[23])+','+str(row[24])+','+str(row[25])+','+str(row[26])+','+str(row[27])+','+str(row[28])+','+str(row[29])+','+str(row[30])+','+str(row[31])+','+str(row[32])+','+str(row[33])+','+str(row[34])+','+str(row[35])+','+str(row[36])+','+str(row[37])+','+str(row[38])+','+str(row[39])+','+str(row[40])+','+str(row[41])+','+str(row[42])+','+str(row[43])+','+str(row[44])+','+str(row[45])+','+str(row[46])+','+str(row[47])+','+str(row[48])+','+str(row[49])+','+str(row[50])+','+str(row[51])+','+str(row[52])+','+str(row[53])+','+str(row[54])+','+str(row[55])+','+str(row[56])+','+str(row[57])+','+str(row[58])+','+str(row[59])+','+str(row[60])+','+str(row[61])+','+str(row[62])+','+str(row[63])+','+str(row[64])+','+str(row[65])+','+str(row[66])+','+str(row[67])+','+str(row[68])+'\n'
			write_file.write(temp_row)

def Random_forest_v():
	
	dataset = pd.read_csv("train_v.csv")
	dataset_tst =pd.read_csv("test_v.csv")
	print(dataset)
	X_train = dataset.iloc[:,4:70].values  
	y_train = dataset.iloc[:,1].values  
	X_test = dataset_tst.iloc[:, 4:70].values 
	y_test = dataset_tst.iloc[:, 1].values 
	regressor = RandomForestClassifier(n_estimators=203,criterion='gini', random_state=0,bootstrap=True, class_weight=None,
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_jobs=None,
            oob_score=True, verbose=0, warm_start=False)  
	regressor.fit(X_train, y_train)  
	y_pred = regressor.predict(X_test) 
	conf_matrix=confusion_matrix(y_test,y_pred)  
	print(conf_matrix)
	#print(accuracy_score(y_test, y_pred))  
	precision,recall,fscore,support= precision_recall_fscore_support(y_test,y_pred)
	conf_matrix=np.array(conf_matrix)
	zero_row=np.where(~conf_matrix.any(axis=1))[0]
	'''for i in range (len(conf_matrix)):
		if i  not in zero_row:
			if(np.argmax(conf_matrix[i])==i):
				return 'correct user'
			else:
				return 'incorrect user'
				'''
	conf_shape=np.shape(conf_matrix)

	'''for i in range (len(conf_matrix)):
		if i  not in zero_row:
			if(np.argmax(conf_matrix[i])==i):
				return 'correct user'
			else:
				return 'incorrect user'''


	for i in range (len(conf_matrix)):
		if i  not in zero_row:
			perc=conf_matrix[i][i]/(((support[i]-conf_matrix[i][i])/(len(conf_matrix)-1))+conf_matrix[i][i])
			print(perc*100,'percentage.............................')
			return perc*100
					