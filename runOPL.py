import matplotlib.pyplot as plt
import os
import random
from numpy import linalg as LA
import numpy as np
from sklearn import preprocessing
from scipy import stats
from scipy.spatial.distance import cdist
import PAA
import PLAlgorithms
import GetDataset

#SYNSEP_CATEGORY_MAPPING = ['0', '1', '2', '3', '4', '5', '6', '7'] #ecoli
#SYNSEP_CATEGORY_MAPPING = ['0', '1', '2', '3', '4', '5']                    #dermatology
#SYNSEP_CATEGORY_MAPPING = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] #usps
#SYNSEP_CATEGORY_MAPPING = ['0', '1', '2', '3', '4', '5']  #satimage
#SYNSEP_CATEGORY_MAPPING = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  #mnist
#SYNSEP_CATEGORY_MAPPING = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']  #letter

GD=GetDataset.Dataset('cifar-10')

features,labels=GD.loadData()

#features=preprocessing.scale(features) #preprocessing step

q=3 #number of extra labels in labels set
T=10000 #number of iterations
[m,d]=features.shape #m=number of instances, d=dimension
L=len(np.unique(labels))

c_value = [0.001]

error_list=np.zeros([])

al = int(T/100)

er = [0.0 for x in range(0,al)]
er1 = [0.0 for x in range(0,al)]
er2 = [0.0 for x in range(0,al)]
er3 = [0.0 for x in range(0,al)]
er4 = [0.0 for x in range(0,al)]
er5 = [0.0 for x in range(0,al)]
er6 = [0.0 for x in range(0,al)]
er7 = [0.0 for x in range(0,al)]

	
rr = [0 for x in range(0,al)]

c1=0.0001
c2=0.005
c3=0.005
lambda1=0.1
lambda2=0.1

epochs=10 #total number of epochs to run code

error_vals=np.zeros([7,epochs,al])

feat = []
feat_label = []
feat_label_set = []

for var in range(0,len(c_value)):
	for k in range(0,epochs):
		print("Run: %d" %int(k))
	
		indices=np.random.choice(m,T,replace=True)
		partial_label_list=list()
	
		#np.random.seed(seed=1771)
		for i in range(m):

			remaining_labels=np.delete(np.arange(L),labels[i])
			extra_labels=np.random.choice(remaining_labels, q, replace=False)
			extra_labels=np.append(extra_labels,labels[i])
			partial_label_list.append(extra_labels)
			
			palpa = PAA.PalPA(d, c_value[var], L)
			palpa1 = PAA.PalPA1(d, c2, L)
			palpa2 = PAA.PalPA2(d, c3, L)
		
			avgPerceptron = PLAlgorithms.AveragePredictionPerceptron(d, L)
			maxPerceptron = PLAlgorithms.MaxPredictionPerceptron(d, L)
			avgPegasos = PLAlgorithms.AveragePredictionPegasos(d, lambda1, L)
			maxPegasos = PLAlgorithms.MaxPredictionPegasos(d, lambda2, L)
			
		error_list = list()
		error_list1 = list()
		error_list2 = list()
		error_list3 = list()
		error_list4 = list()
		error_list5 = list()
		error_list6 = list()
		

		rounds = list()

		
		for t in range(0,T):
			if var==0:
				feature_vectors, true_label, label_set = (features[indices[t],:]).tolist(),labels[indices[t]],partial_label_list[indices[t]]
				feat.append(feature_vectors)
				feat_label.append(true_label)
				feat_label_set.append(label_set)

			
			else:
				feature_vectors = feat[t]
				true_label = feat_label[t]
				label_set = feat_label_set[t]

			palpa.run(feature_vectors, true_label, label_set)
			palpa1.run(feature_vectors, true_label, label_set)
			palpa2.run(feature_vectors, true_label, label_set)
			
			avgPerceptron.run(features[indices[t],:], true_label, label_set)
			maxPerceptron.run(features[indices[t],:], true_label, label_set)
			avgPegasos.run(features[indices[t],:], true_label, label_set)
			maxPegasos.run(features[indices[t],:], true_label, label_set)
				
			if ((t+1)%100) == 0:
				print("%s rounds completed with error rate %s by PA" %(str(t+1),str(palpa.error_rate)))
				print("%s rounds completed with error rate %s by PA-I" %(str(t+1),str(palpa1.error_rate)))
				print("%s rounds completed with error rate %s by PA-II" %(str(t+1),str(palpa2.error_rate)))

				print("%s rounds completed with error rate %s by Avg Perceptron" %(str(t+1),str(avgPerceptron.error_rate)))
				print("%s rounds completed with error rate %s by Max Perceptron" %(str(t+1),str(maxPerceptron.error_rate)))
				print("%s rounds completed with error rate %s by Avg Pegasos" %(str(t+1),str(avgPegasos.error_rate)))
				print("%s rounds completed with error rate %s by Max Pegasos" %(str(t+1),str(maxPegasos.error_rate)))

				rounds.append(palpa.number_of_rounds)

				error_list.append(palpa.error_rate)
				error_list1.append(palpa1.error_rate)
				error_list2.append(palpa2.error_rate)
				error_list3.append(avgPerceptron.error_rate)
				error_list4.append(maxPerceptron.error_rate)
				error_list5.append(avgPegasos.error_rate)
				error_list6.append(maxPegasos.error_rate)
				
				print("=================================")
					
				 
		for i in range(0,al):
			er[i] += error_list[i]
			er1[i] += error_list1[i]
			er2[i] += error_list2[i]
			er3[i] += error_list3[i]
			er4[i] += error_list4[i]
			er5[i] += error_list5[i]
			er6[i] += error_list6[i]
			
			
			error_vals[0,k,i] = error_list[i]
			error_vals[1,k,i] = error_list1[i]
			error_vals[2,k,i] = error_list2[i]
			error_vals[3,k,i] = error_list3[i]
			error_vals[4,k,i] = error_list4[i]
			error_vals[5,k,i] = error_list5[i]
			error_vals[6,k,i] = error_list6[i]			

			
	for i in range(0,al):
		rr[i] = 100*(i+1)
			
	er=[x/epochs for x in er]
	er1=[x/epochs for x in er1]
	er2=[x/epochs for x in er2]
	er3=[x/epochs for x in er3]
	er4=[x/epochs for x in er4]
	er5=[x/epochs for x in er5]
	er6=[x/epochs for x in er6]
	
a = {'pa': er, 'pa1': er1, 'pa2': er2, 'avgpc': er3, 'maxpc': er4, 'avgpg': er5, 'maxpg': er6, 'rounds': rr}
datavalall = open('abc.txt','w+')
datavalall.write(str(a))
datavalall.close()