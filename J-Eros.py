'''
    Copyright (C)Taban Eslami and Fahad Saeed
    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 2
    of the License, or (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
'''

from __future__ import division
import glob
def findFiles(path): return sorted(glob.glob(path))
import numpy as np
import random
np.random.seed(0)
import math
import statistics
import sys
from numpy import dot
from numpy import inner
from numpy.linalg import norm
import math
import collections
import itertools
from scipy.stats import mode
from sklearn.model_selection import KFold
import time
############################################################################
traindata=[]
testdata=[]
###reading data from file
for filename in findFiles('/home/taban/Eros/Data2/KKI-train/*'):
 
    label=int(filename[-1])
    print(filename)
    for subfilename in findFiles(filename+'/*/sfn*rest_1*.1D'):
        print(subfilename)
        aa=np.genfromtxt(subfilename,dtype=None, delimiter='\t', names=True)
        d =aa.dtype.names
        regionnum=len(d)-2
        
        a=np.genfromtxt(subfilename,dtype=None,skip_header=1,delimiter="\t")#,delimiter=",  ")
        b = np.zeros((len(aa),(regionnum)))
        for i in range(0,len(aa)):
            for j in range(0,(regionnum)):
                b[i][j]=a[i][j+2]
        traindata.append([np.array(b).T,label])

for filename in findFiles('/home/taban/Eros/Data2/KKI-test/*'):
    
    label=int(filename[-1])
    print(filename)
    for subfilename in findFiles(filename+'/*/sfn*rest_1*.1D'):
        aa=np.genfromtxt(subfilename,dtype=None, delimiter='\t', names=True)
        d =aa.dtype.names
        regionnum=len(d)-2
        
        a=np.genfromtxt(subfilename,dtype=None,skip_header=1,delimiter="\t")#,delimiter=",  ")
        b = np.zeros((len(aa),(regionnum)))
        for i in range(0,len(aa)):
            for j in range(0,(regionnum)):
                b[i][j]=a[i][j+2]
        #traindata.append([b,label,filename[-3]])
        testdata.append([np.array(b).T,label])


traindata=np.array(traindata)
print("train data shape",traindata.shape)
testdata=np.array(testdata)
print("test data shape",testdata.shape)

traindatacut=[]
for sub in range(len(traindata)):
    cut=[]
    b=(traindata[sub][0])[:,10:]
    traindatacut.append([np.array(b),traindata[sub][1]])

traindatacut=np.array(traindatacut)

testdatacut=[]
for sub in range(len(testdata)):
    cut=[]
    b=(testdata[sub][0])[:,10:]
    testdatacut.append([np.array(b),testdata[sub][1]])

testdatacut=np.array(testdatacut)

print("traindatacut.shape",traindatacut.shape)
print("testdatacut.shape",testdatacut.shape)
print(testdatacut[0][0].shape)
print(traindatacut[0][0].shape)

##########################################################
def cosine_sim(a,b):
    
   # cos_sim = inner(a, b)/(norm(a)*norm(b))
    return abs(np.inner(a,b))#abs(cos_sim)

def compute_weigth(training,region_num):
    S=[]
    w=[]
    summm=0

   ####normalizing each eigen_value_vector seperately
    for sub_iter in range(training.shape[0]):
        eigen_values = training[sub_iter][0].copy()

        summm=np.sum(eigen_values)

        for j_iter in range(len(eigen_values)):
            eigen_values[j_iter]=eigen_values[j_iter]/summm

        S.append(eigen_values)


    S=np.array(S)

    summ=0

    for i_iter in range(region_num):
        w.append(np.mean(S[:,i_iter]))
        summ=summ+np.mean(S[:,i_iter])#W[i_iter]

    for i_iter in range(region_num):
        w[i_iter]=w[i_iter]/summ

    w=np.array(w)

    return (w)



region_num = 190

def Eros(sub_test_index,sub_train_index,eigen_vecs_vals,eigen_vecs_vals_test,W):
    
    eig_vecs1=eigen_vecs_vals[sub_train_index]
    eig_vecs2=eigen_vecs_vals_test[sub_test_index]
    summ=0
    for ie in range(region_num):#for over number of regions
        summ=summ+(W[ie]*cosine_sim(eig_vecs1[ie],eig_vecs2[ie]))

    return summ

##########################

def knn_Eros(train,test,k,W):
    count_dorost=0
    count_healthy=0#predicted healthy correct
    count_adhd=0#predicted adhd correct
    total_predicted_adhd=0
    healthy=0
    adhd=0
    trainlabel=train[:,2]
    testlabel=test[:,2]



    eigen_vecs_train=train[:,1]
    eigen_vecs_test=test[:,1]


    for i in range(len(test)):
        if(testlabel[i]==0):
            healthy=healthy+1
        if(testlabel[i]!=0):
            adhd=adhd+1



        eros_dis=np.ones(len(train))#list o distance of all training subjects to s test subject
        for j in range(len(train)):
            sim=Eros(i,j,eigen_vecs_train,eigen_vecs_test,W)
            
            eros_dis[j]=sim
        kkg=k*(-1)
        k_neighbors=np.argsort(np.array(eros_dis))[kkg:]
            
        knn_labels=trainlabel[k_neighbors]

        mode_data = mode(np.array(knn_labels), axis=0)
        mode_label = mode_data[0]
        if mode_data[0][0]!=0:
            total_predicted_adhd=total_predicted_adhd+1

        if mode_data[0][0]==testlabel[i]:
            count_dorost=count_dorost+1
            if mode_data[0][0]==0:
                count_healthy=count_healthy+1

            if mode_data[0][0]!=0:
                count_adhd=count_adhd+1

        if mode_data[0][0]!=testlabel[i] and mode_data[0][0]!=0 and testlabel[i]!=0:
            count_adhd=count_adhd+1
                        

    if adhd!=0:
        sensitivity=count_adhd/adhd
    else:
        print("------there is no adhd subject in test sub sample-------")
        print(testlabel)
        print("-------------")
        sensitivity=-1

    if healthy!=0:
        specificity=count_healthy/healthy
    else:
        specificity=-1

    total_acc=count_dorost/len(test)
    return total_acc,sensitivity,specificity,sensitivity+specificity


####training preprocessing

train_all_eigen_vals_vecs_labels=[]

for j in range(len(traindatacut)):
    sub=traindatacut[j][0]
    label=traindatacut[j][1]
    eig_vals1, eig_vecs1 = np.linalg.eig(np.cov(sub))
    eig_pairs1 = [(eig_vals1[i], eig_vecs1[:,i]) for i in range(len(eig_vals1))]
    eig_pairs1 = [(eig_vals1[i], eig_vecs1[:,i]) for i in range(len(eig_vals1))]
    eig_pairs1.sort(key=lambda x: x[0], reverse=True)

    temp_val=[]
    temp_vec=[]
    for ii in range(len(eig_pairs1)):
        temp_val.append(np.array(eig_pairs1[ii][0]))
        temp_vec.append(np.array(eig_pairs1[ii][1]))
    train_all_eigen_vals_vecs_labels.append([temp_val,temp_vec,label])

train_all_eigen_vals_vecs_labels=np.array(train_all_eigen_vals_vecs_labels)


####test preprocessing
test_all_eigen_vals_vecs_labels=[]
for j in range(len(testdatacut)):
    sub=testdatacut[j][0]
    label=testdatacut[j][1]
    eig_vals1, eig_vecs1 = np.linalg.eig(np.cov(sub))
    eig_pairs1 = [(eig_vals1[i], eig_vecs1[:,i]) for i in range(len(eig_vals1))]
    eig_pairs1.sort(key=lambda x: x[0], reverse=True)
    
    temp_val=[]
    temp_vec=[]
    for ii in range(len(eig_pairs1)):
        temp_val.append(np.array(eig_pairs1[ii][0]))
        temp_vec.append(np.array(eig_pairs1[ii][1]))
    test_all_eigen_vals_vecs_labels.append([temp_val,temp_vec,label])

test_all_eigen_vals_vecs_labels=np.array(test_all_eigen_vals_vecs_labels)


##########################

start=time.time()
TF=np.zeros(10)
sens_plus_spefs=np.zeros(10)
sens_koll=np.zeros(10)
spef_koll=np.zeros(10)
avg_sens=np.zeros(10)
avg_spef=np.zeros(10)
J_stat=np.zeros(10)
train_copy=train_all_eigen_vals_vecs_labels.copy()
fold_num=4


for k in range(1,11):#[9,10]:# range(1,11):
    print("This is k: ",k)
    np.random.seed(0)
    train_copy=train_all_eigen_vals_vecs_labels[train_all_eigen_vals_vecs_labels[:,2].argsort()].copy()
    for tekrar in range(10):
        np.random.shuffle(train_copy)
        kf = KFold(n_splits=fold_num)
        sens_neg=0
        spef_neg=0
        total_acc=0
        total_sens=0
        total_spef=0
        
        for train_index, test_index in kf.split(train_copy):
            traindatacut_cv=train_copy[train_index]
            testdatacut_cv=train_copy[test_index]
            W=compute_weigth(traindatacut_cv,region_num)
            acc,sens,spef,sens_spef=knn_Eros(traindatacut_cv,testdatacut_cv,k,W)
            total_acc=total_acc+acc
            
            if sens!=-1:
                total_sens=total_sens+sens
            else:
                sens_neg=sens_neg+1
                print("no snes for this fold")
            
            if spef!=-1:
                total_spef=total_spef+spef
            else:
                spef_neg=spef_neg+1
                print("no spef for this fold")

        z_acc=(total_acc/fold_num)
        if fold_num-spef_neg!=0 and fold_num-sens_neg!=0:
            z_sens=(total_sens/(fold_num-sens_neg))
            z_spef=(total_spef/(fold_num-spef_neg))

        else:
            print("warning, none of folds have sensitivity ",fold_num-sens_neg,fold_num-spef_neg)
#        print("\n",k, z_acc, z_sens, z_spef, z_sens+z_spef)
        sens_plus_spefs[k-1]=sens_plus_spefs[k-1]+(z_sens+z_spef)
        sens_koll[k-1]=sens_koll[k-1]+z_sens
        spef_koll[k-1]=spef_koll[k-1]+z_spef
        if z_sens==0 or z_spef==0:
            TF[k-1]=1

for tekrar2 in range(10):
    avg_sens[tekrar2]=sens_koll[tekrar2]/10
    avg_spef[tekrar2]=spef_koll[tekrar2]/10
    if TF[tekrar2]==0:
        J_stat[tekrar2]=avg_sens[tekrar2]+avg_spef[tekrar2]-1
    else:
        print("k = ",tekrar2+1,"is ignored since it had sensitivity/specificity = 0 in one of rounds")
        J_stat[tekrar2]=-10000

print("In the following array 1 shows the k that shouldn't be considered (had sensitiity or sepcifiticy equal to 0)")
print(TF)
print("\n Sensitivity of each k:")

for i in range(10):
    print("Sensitivity of k = ",i," is ",sens_koll[i])

print("--------------------------------------------------\n Sensitivity of each k:")

for i in range(10):
    print("specificity of k = ",i," is ",spef_koll[i])

opt_k=np.argmax(J_stat)+1
print("--------------------------------------------------\nOptimal valueof k is: ",opt_k)
wabel=compute_weigth(train_all_eigen_vals_vecs_labels,region_num)
acc,adhd,health,health_adhd=knn_Eros(train_all_eigen_vals_vecs_labels,test_all_eigen_vals_vecs_labels,opt_k,wabel)    
print("Result of testing on K= ",opt_k," is: ")
print("accuracy: ",acc,"\n","sensitivity: ",adhd,"specificity: ",health)

end=time.time()


print("Running time is: :",end-start)

