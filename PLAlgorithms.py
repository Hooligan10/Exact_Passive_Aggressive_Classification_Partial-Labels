import numpy as np
from numpy import linalg as LA

class AveragePredictionPerceptron:
	
	def __init__(self,dim=7,L=8):
		self.dict_length = dim #dimension
		self.number_of_classes = L
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0
		self.label_list=np.asarray([i for i in range(self.number_of_classes)])
	
	def init_weights(self):
		return np.zeros([self.number_of_classes,self.dict_length])
	

	def predict_scores(self, feature_vectors):
		
		## feature vector is a numy array
		x=feature_vectors.reshape(self.dict_length,1)
		scores=np.matmul(self.weights,x)
		
		return scores

	def update_weights(self,scores, feature_vectors, true_label):
		
		x=feature_vectors.reshape(self.dict_length,1)
		Y=np.asarray(true_label)
		Y=np.sort(Y)
		Y_bar=np.array([j for j in self.label_list if j not in Y])
		pl_scores=scores[Y]
		nonpl_scores=scores[Y_bar]
		chk=(1/(1.0*len(Y)))*np.sum(pl_scores)-np.max(nonpl_scores)
		
		if chk<1:
			for j in Y:
				self.weights[j,:]=self.weights[j,:]+(x.T/(1.0*len(Y)))
			
			maxscore_c_pos=np.argmax(nonpl_scores)
			y_bar_max=Y_bar[maxscore_c_pos]
			self.weights[y_bar_max,:]=self.weights[y_bar_max,:]-x.T

	def run(self, feature_vectors, true_label, label_set):
		
		self.number_of_rounds += 1.0
		scores = self.predict_scores(feature_vectors)
		predicted_label=np.argmax(scores)
		
		if predicted_label==true_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		
		self.update_weights(scores, feature_vectors, label_set)
		

			
class MaxPredictionPerceptron:
	
	def __init__(self,dim=7,L=8):
		
		self.dict_length = dim #dimension
		self.number_of_classes = L
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0
		self.label_list=np.asarray([i for i in range(self.number_of_classes)])
	
	def init_weights(self):
		return np.zeros([self.number_of_classes,self.dict_length])
	
	def predict_scores(self, feature_vectors):
		
		## feature vector is a numy array
		x=feature_vectors.reshape(self.dict_length,1)
		scores=np.matmul(self.weights,x)
		
		return scores

	def update_weights(self,scores, feature_vectors, true_label):
		
		x=feature_vectors.reshape(self.dict_length,1)
		Y=np.asarray(true_label)
		Y=np.sort(Y)
		Y_bar=np.array([j for j in self.label_list if j not in Y])
		pl_scores=scores[Y]
		nonpl_scores=scores[Y_bar]
		chk=np.max(pl_scores)-np.max(nonpl_scores)
		
		if chk<1:
			maxscore_Y_pos=np.argmax(pl_scores)
			y_max=Y[maxscore_Y_pos]
			self.weights[y_max,:]=self.weights[y_max,:]+x.T
		
			maxscore_Yc_pos=np.argmax(nonpl_scores)
			y_bar_max=Y_bar[maxscore_Yc_pos]
			self.weights[y_bar_max,:]=self.weights[y_bar_max,:]-x.T

	def run(self, feature_vectors, true_label, label_set):
		
		self.number_of_rounds += 1.0
		scores = self.predict_scores(feature_vectors)
		predicted_label=np.argmax(scores)
		
		if predicted_label==true_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		
		self.update_weights(scores, feature_vectors, label_set)
	


class AveragePredictionPegasos:
	
	def __init__(self,dim=7,lambda_const=0.001,L=8):
		self.lambda_const=lambda_const
		self.dict_length = dim #dimension
		self.number_of_classes = L
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0
		self.label_list=np.asarray([i for i in range(self.number_of_classes)])
	
	def init_weights(self):
		return np.zeros([self.number_of_classes,self.dict_length])
	
	def predict_scores(self, feature_vectors):
		
		## feature vector is a numy array
		x=feature_vectors.reshape(self.dict_length,1)
		scores=np.matmul(self.weights,x)
		
		return scores

	def update_weights(self,scores, feature_vectors, true_label):
		
		x=feature_vectors.reshape(self.dict_length,1)
		Y=np.asarray(true_label)
		Y=np.sort(Y)
		Y_bar=np.array([j for j in self.label_list if j not in Y])
		pl_scores=scores[Y]
		nonpl_scores=scores[Y_bar]
		chk=(1/(1.0*len(Y)))*np.sum(pl_scores)-np.max(nonpl_scores)
		eta=1.0/(self.lambda_const*(self.number_of_rounds))
		
		self.weights=(1.0-(1.0/(1.0*self.number_of_rounds)))*self.weights
		
		if chk<1:
			for j in Y:
				self.weights[j,:]=self.weights[j,:]+eta*(x.T/(1.0*len(Y)))
			
			maxscore_c_pos=np.argmax(nonpl_scores)
			y_bar_max=Y_bar[maxscore_c_pos]
			self.weights[y_bar_max,:]=self.weights[y_bar_max,:]-eta*x.T
		
		W_norm=LA.norm(self.weights)
		mult_const=1.0
		
		if mult_const>((1.0)/(np.sqrt(self.lambda_const)*W_norm)):
			mult_const=(1.0)/(np.sqrt(self.lambda_const)*W_norm)
		
		self.weights=mult_const*self.weights
		

	def run(self, feature_vectors, true_label, label_set):
		
		self.number_of_rounds += 1.0
		scores = self.predict_scores(feature_vectors)
		predicted_label=np.argmax(scores)
		
		if predicted_label==true_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		
		self.update_weights(scores, feature_vectors, label_set)
		
   
	
class MaxPredictionPegasos:
	
	def __init__(self,dim=7,lambda_const = 0.001,L=8):
		self.lambda_const = lambda_const
		self.dict_length = dim #dimension
		self.number_of_classes = L
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0
		self.label_list=np.asarray([i for i in range(self.number_of_classes)])
	
	def init_weights(self):
		return np.zeros([self.number_of_classes,self.dict_length])
	
	def predict_scores(self, feature_vectors):
		
		## feature vector is a numy array
		x=feature_vectors.reshape(self.dict_length,1)
		scores=np.matmul(self.weights,x)
		
		return scores

	def update_weights(self,scores, feature_vectors, true_label):
		
		x=feature_vectors.reshape(self.dict_length,1)
		Y=np.asarray(true_label)
		Y=np.sort(Y)
		Y_bar=np.array([j for j in self.label_list if j not in Y])
		pl_scores=scores[Y]
		nonpl_scores=scores[Y_bar]
		chk=np.max(pl_scores)-np.max(nonpl_scores)
		
		eta=1.0/(self.lambda_const*(self.number_of_rounds))
	
		self.weights=(1.0-(1.0/(self.number_of_rounds)))*self.weights
		
		if chk<1:
			maxscore_Y_pos=np.argmax(pl_scores)
			y_max=Y[maxscore_Y_pos]
			self.weights[y_max,:]=self.weights[y_max,:]+eta*x.T
		
			maxscore_Yc_pos=np.argmax(nonpl_scores)
			y_bar_max=Y_bar[maxscore_Yc_pos]
			self.weights[y_bar_max,:]=self.weights[y_bar_max,:]-eta*x.T
		
		W_norm=LA.norm(self.weights)
		mult_const=1.0
		
		if mult_const>((1.0)/(np.sqrt(self.lambda_const)*W_norm)):
			mult_const=(1.0)/(np.sqrt(self.lambda_const)*W_norm)
		
		self.weights=mult_const*self.weights
	
	def run(self, feature_vectors, true_label, label_set):
		
		self.number_of_rounds += 1.0
		scores = self.predict_scores(feature_vectors)
		predicted_label=np.argmax(scores)
		
		if predicted_label==true_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		
		self.update_weights(scores, feature_vectors, label_set)
