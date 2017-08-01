from __future__ import print_function
import numpy as np
import os
import scipy.io as sio
import cPickle as pickle
import re
import cv2
import itertools as it

data_path = '/media/songguoxian/b1efdde7-81bb-4db1-897b-b5ff506288bc/songguoxian/Dataset/EyeRegionReal/'
label_path = '/media/songguoxian/New Volume/Guoxian/Dropbox (NTU)/NTU AU/ActionUnit_Labels/'
image_width = 72
image_height= 54
pixel_depth = 255

def Read_label():   # save to mat 23
    FullLabelMat=[]
    for i in it.chain(range(3,13),range(16,19),range(21,22),range(23,30),range(31,33)):
        FullLabelMat.append(LoadAu(i))
    return FullLabelMat

def AuLabel(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split(',')
        curLine = curLine[1]
        intLine = map(int, curLine)  # map all elements to float()
        dataMat.append(intLine[0])
    return dataMat

def LoadAu(subject):
    path = label_path
    if subject < 10:
        path = path + 'SN00' + str(subject) + '/'
    else:
        path = path + 'SN0' + str(subject) + '/'

    AU = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]

    if subject < 10:
        filename = path + 'SN00' + str(subject)
    else:
        filename = path + 'SN0' + str(subject)

    tmpMat = np.array(AuLabel(filename + '_au1.txt'))
    dataMat = np.zeros((tmpMat.size, 12))
    dataMat[:, 0] = tmpMat
    for i in range(1, 11):
        tmpMat = np.array(AuLabel(filename + '_au' + str(AU[i + 1]) + '.txt'))
        dataMat[:, i] = tmpMat
    return dataMat


def MakeDataset(FullLabelMat,rootDir):
    list_dirs = os.walk(rootDir)
    dataset_img =[]
    dataset_label=[]
    Dic = BuildDic()
    cout=0
    for root, dirs,files in list_dirs:
        subject = root[len(rootDir):]
        cout+=1
        for f in files:
            print(root + f)
            img = cv2.imread(os.path.join(root,f))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            img=np.reshape(img,(1,image_width*image_height))
            if dataset_img==[]:
                dataset_img=img;
                dataset_label= np.array([int(FullLabelMat[Dic[subject]][int(re.split('L|R|.png',f)[1]),0] )])
            else:
                dataset_img=np.append(dataset_img,img,axis=0)
                dataset_label=np.append(dataset_label,[int(FullLabelMat[Dic[subject]][int(re.split('L|R|.png',f)[1]),0])]
                                        ,axis=0)

    dataset_label=np.reshape(dataset_label,(len(dataset_label),1))
    return dataset_label,dataset_img

def BuildDic():
    Dic={}
    cout=0;
    for i in it.chain(range(3, 13), range(16, 19), range(21, 22), range(23, 30), range(31, 33)):
        Dic[str(i)]=cout
        cout+=1
    return Dic

def Randomzie(dataset_label,dataset_img):
    permutation = np.random.permutation(dataset_label.shape[0])
    shuffled_label = dataset_label[permutation]
    shuffled_img = dataset_img[permutation,:]
    return shuffled_label, shuffled_img


FullLabelMat= Read_label()
dataset_label,dataset_img=MakeDataset(FullLabelMat,data_path)


Randomzie(dataset_label,dataset_img)
print(np.shape(dataset_label),np.shape(dataset_img))

pickle_file = 'dataset.pickle'
try:
  f = open(pickle_file, 'wb')
  save = {
    'label': dataset_label,
    'img': dataset_img,
    }
  pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', pickle_file, ':', e)
  raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)

sdadsfasd=1