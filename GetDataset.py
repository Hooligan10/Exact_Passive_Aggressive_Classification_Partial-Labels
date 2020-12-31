import numpy as np
import pandas as pd
import h5py
from sklearn import preprocessing
import scipy.io


class Dataset:
	def __init__(self, dataset_name):
		self.dataset_name = dataset_name

	def loadData(self):
		name = (self.dataset_name).lower()
		
		if name == 'ecoli':            
			data=pd.read_csv("uci_ecoli.csv",header=None)
			data=data.values
			data=data[:,1:9]
			le = preprocessing.LabelEncoder()
			le.fit(data[:,7])
			labels=le.transform(data[:,7])
			X=data[:,0:7]
			X = X.astype(float)
		
		elif name == 'dermatology':
			data=pd.read_csv("dermatology.txt",header=None)
			data=data.values
			labels=data[:,34]-1
			X=data[:,0:34]
			X = X.astype(float)
			
		
		elif name == 'satimage':
			data_train=pd.read_csv("sat.trn",header=None,sep=' ')
			data_train=data_train.values
			
			data_test=pd.read_csv("sat.tst",header=None,sep=' ')
			data_test=data_test.values
			X_train=data_train[:,0:36]
			X_test=data_test[:,0:36]
			
			X = (np.concatenate((X_train,X_test)))
			X = X.astype(float)
			train_labels = data_train[:,36]
			test_labels = data_test[:,36]
			
			labels=np.concatenate((train_labels,test_labels))
			labels[labels==7]=6
			labels=labels-1
		
		elif name == 'usps':
			filename = 'usps.h5'
			f = h5py.File(filename, 'r')
			train_group = f['train']
			test_group = f['test']
			train_data=train_group['data'].value
			test_data=test_group['data'].value
			train_labels=train_group['target'].value
			test_labels=test_group['target'].value
			X=(np.concatenate((train_data,test_data)))
			labels=np.concatenate((train_labels,test_labels))
			f.close()
		
		elif name == 'mnist':
			mat = scipy.io.loadmat('./mnist.mat')
			X=mat['A']
			labels=mat['L']

		elif name == 'cifar-10':
			data = np.load("./CIFAR10_{}-keras_features.npz".format('vgg16'))
			train_data = data['features_training']
			train_labels = data['labels_training']
			test_data = data['features_testing']
			test_labels = data['labels_testing']
			
			X = np.concatenate((train_data, test_data))
			labels = np.concatenate((train_labels, test_labels))
		
		else:
			print('Dataset not found')
			X=[]
			labels=[]
			
		return X,labels

		
		

