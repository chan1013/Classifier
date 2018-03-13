from __future__ import print_function
import os
from tkinter import * 
from tkinter import ttk
from tkinter.filedialog import askdirectory
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants
from os import listdir
from os.path import isfile, isdir, join
import scipy
import PIL
from PIL import Image

def ImageFileToArray(path):
	Folders = listdir(path)
	FileName=[]#各資料夾內檔名
	ImageData=[]
	#ImageData=np.zeros((0,0))
	for i,folder in enumerate(Folders):
		folderpath = join(path, folder)#子目錄路徑
		filename_temp=np.asarray(listdir(folderpath))#子目錄內檔案
		FileName.append(filename_temp)
		for j,img in enumerate(filename_temp):#每個檔案存成array
			ImgPath=join(folderpath, img)
			ImgArr=ImageToArr(ImgPath)
			ImgArr=np.asarray(ImgArr)
			ImageData.append(ImgArr)
			#np.concatenate(ImageData,ImgArr)
	FileName=np.asarray(FileName)
	ImageData=np.asarray(ImageData)
	
	return FileName,ImageData
	
def ImageToArr(path):
	im = Image.open(path)
	data = im.getdata()
	data = np.matrix(data)
	im.close()
	return data

#照各類別筆數給一個label矩陣	
def make_label(data):
	label=np.zeros((1,data.shape[0]))
	for i in range(data.shape[0]):
		temp=np.zeros((data[i].shape[0],data.shape[0]))
		temp[:,i]=1
		label=np.concatenate((label,temp),axis=0)
	return label[1:,:]
	
def Onehot(Arrary):
	label=np.zeros((Arrary.shape[0],np.max(Arrary)+1))
	for i in range(Arrary.shape[0]):
		index=Arrary[i]
		label[i,index]=1
	return label
