class PalPA:
	def __init__(self, dim=7, gamma=0.02, L=8):
		self.gamma = 0.02
		self.dict_length = dim
		self.number_of_classes = L
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0
		

	def init_weights(self):
		weights = []
		for i in range(0,self.number_of_classes):
			weights.append([0.0] * self.dict_length)
		
		return weights

	def predict_label(self, feature_vectors):
		maxi = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]

			if total >= maxi:
				maxi = total
				label = i
		
		return label


	def update_weights(self, update_matrix):
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]

	def get_update_matrix(self, feature_vectors, predicted_label, label_set, support_set,l_tilde,labeltil):
		update_matrix = self.init_weights()
		step_lambda = [0.0 for x in range(0,len(update_matrix))] ## Lambdas for each label
		
		total_support_class_loss = 0.0
		
		modxsq = 0.0
		sum_step_lambda = 0.0
		
		for i in range(0,len(support_set)):
			total_support_class_loss += l_tilde[support_set[i]]		

		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		### Determining lambdas for each support class label, non support class has lambda = 0
		for r in range(0,len(support_set)):
			step_lambda[support_set[r]] = (l_tilde[support_set[r]] - (total_support_class_loss/(len(support_set) + len(label_set))))/modxsq
		
		### Calculating the sum of all lambdas
		for j in range(0,len(step_lambda)):
			sum_step_lambda += step_lambda[j]

		### Determining the weight update corresponding to each label
		for i in range(0,len(update_matrix)):
			if i in true_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = feature_vectors[j] * (sum_step_lambda/len(label_set))
			else:                
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = -feature_vectors[j] * (step_lambda[i])


		return update_matrix

	def run(self, feature_vectors, true_label, label_set):
		self.number_of_rounds += 1.0
		predicted_label = self.predict_label(feature_vectors)

		if predicted_label==true_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		

		### Calculating l_tilde for each label
		l_tilde = [-1.0 for x in range(len(self.weights))]
		lt_tilde = [-1.0 for x in range(len(self.weights))]
		labeltil = [-1 for x in range(len(self.weights))]
		label_set_total = 0.0

		for r in range(0,len(label_set)):
			for eachVector in range(0,len(feature_vectors)):
				label_set_total +=  feature_vectors[eachVector]*self.weights[r][eachVector]
		
		label_set_total = label_set_total/len(label_set)
		
		j = 0
		for r in range(0,len(self.weights)):
			if r not in label_set:
				loss_r = 0.0
				for eachVector in range(0,len(feature_vectors)):
					loss_r += feature_vectors[eachVector]*self.weights[r][eachVector]
				l_tilde[r] = max(0.0, 1 - label_set_total + loss_r)
				lt_tilde[r] = max(0.0, 1 - label_set_total + loss_r)
				labeltil[r] = r

		### Sorting in Decreasing order the l_tilde copy and the corresponding label
		for i in range(len(self.weights)):
			for j in range(0, len(self.weights)-i-1):
				if lt_tilde[j] < lt_tilde[j+1]:
					lt_tilde[j], lt_tilde[j+1] = lt_tilde[j+1], lt_tilde[j]
					labeltil[j], labeltil[j+1] = labeltil[j+1], labeltil[j]

		support_set = []

		cm_loss = 0.0 # cumulative loss


		### Determining Support Classes
		mods = len(self.weights)
		if(lt_tilde[0] != 0.0):
			support_set.append(labeltil[0])
			j = 1

			cm_loss += l_tilde[labeltil[j-1]]
			while(j!= mods):
				if (labeltil[j] != -1 and l_tilde[labeltil[j]] != -1.0):
					rhs = (len(label_set)+j)*l_tilde[labeltil[j]]
					if(cm_loss < rhs):
						support_set.append(labeltil[j])
						cm_loss += l_tilde[labeltil[j]]
						j += 1
					else:
						break
				else:
					break

		# Updating weight matrix using the determined support class set and losses
		update_matrix = self.get_update_matrix(feature_vectors, predicted_label, label_set, support_set,l_tilde,labeltil)
		self.update_weights(update_matrix)

		
class PalPA1:

	def __init__(self, dim=7, c=0.1, L=8):
		self.c=c
		self.dict_length = dim
		self.number_of_classes = L
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		
		weights = []
		for i in range(0,self.number_of_classes):
			weights.append([0.0] * self.dict_length)
		
		return weights

	def predict_label(self, feature_vectors):
		
		maxi = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			if total >= maxi:
				maxi = total
				label = i
		
		return label


	def update_weights(self, update_matrix):
		
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]


	def get_update_matrix(self, feature_vectors, predicted_label, label_set, support_set1,l_tilde,labeltil):
		
		update_matrix = self.init_weights()
		step_lambda1 = [0.0 for x in range(0,len(update_matrix))] ## Lambdas for each label
		total_support_class_loss1 = 0.0
		
		modxsq = 0.0
		sum_step_lambda1 = 0.0
		
		for i in range(0,len(support_set1)):
			total_support_class_loss1 += l_tilde[support_set1[i]]

		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		### Determining lambdas for each support class label, non support class has lambda = 0
		for r in range(0,len(support_set1)):
			tempc = ((total_support_class_loss1)/(modxsq*(1+(len(support_set1)/len(label_set)))))
			step_lambda1[support_set1[r]] = (l_tilde[support_set1[r]]/modxsq) - (total_support_class_loss1/(modxsq*len(support_set1))) + (min(self.c,tempc)/len(support_set1))
		
		### Calculating the sum of all lambdas
		for j in range(0,len(step_lambda1)):
			sum_step_lambda1 += step_lambda1[j]

		### Determining the weight update corresponding to each label
		for i in range(0,len(update_matrix)):
			if i in true_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = feature_vectors[j] * (sum_step_lambda1/len(label_set))
			else:                
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = -feature_vectors[j] * (step_lambda1[i])

		return update_matrix
		

	def run(self, feature_vectors, true_label, label_set):
		
		self.number_of_rounds += 1.0
		predicted_label = self.predict_label(feature_vectors)
		
		if predicted_label == true_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		

		### Calculating l_tilde for each label
		l_tilde = [-1.0 for x in range(len(self.weights))]
		lt_tilde = [-1.0 for x in range(len(self.weights))]
		labeltil = [-1 for x in range(len(self.weights))]
		label_set_total = 0.0
		
		for r in range(0,len(label_set)):
			for eachVector in range(0,len(feature_vectors)):
				label_set_total +=  feature_vectors[eachVector]*self.weights[r][eachVector]
		label_set_total = label_set_total/len(label_set)
		modx=0.0
		j = 0
		
		for j in range(0,len(feature_vectors)):
			modx += feature_vectors[j]*feature_vectors[j]
		for r in range(0,len(self.weights)):
			if r not in label_set:
				loss_r = 0.0

				for eachVector in range(0,len(feature_vectors)):
					loss_r += feature_vectors[eachVector]*self.weights[r][eachVector]
				l_tilde[r] = max(0.0, 1 - label_set_total + loss_r)
				lt_tilde[r] = max(0.0, 1 - label_set_total + loss_r)
				labeltil[r] = r


		### Sorting in Decreasing order the l_tilde copy and corresponding label
		for i in range(len(self.weights)):
			for j in range(0, len(self.weights)-i-1):
				if lt_tilde[j] < lt_tilde[j+1]:
					lt_tilde[j], lt_tilde[j+1] = lt_tilde[j+1], lt_tilde[j]
					labeltil[j], labeltil[j+1] = labeltil[j+1], labeltil[j]


		support_set1 = []
		
		cm_loss1 = 0.0 # cumulative loss
		

		### Determining Support Classes
		mods = len(l_tilde)
		if(lt_tilde[0] != 0.0):
			support_set1.append(labeltil[0])
			j = 1
			cm_loss1 += l_tilde[labeltil[j-1]]
			while(j!= mods):
				if (labeltil[j] != -1 and l_tilde[labeltil[j]] != -1.0):
					tempc = ((cm_loss1)/(modx*(1+(j/len(label_set)))))
					rhs = (j*l_tilde[labeltil[j]]) + (modx*min(self.c,tempc))

					if(cm_loss1 < rhs):
						support_set1.append(labeltil[j])
						cm_loss1 += l_tilde[labeltil[j]]
						j += 1
					else:
						break
				else:
					break

		# Updating weight matrix using the determined support class set and losses
		update_matrix = self.get_update_matrix(feature_vectors, predicted_label, label_set, support_set1,l_tilde,labeltil)
		self.update_weights(update_matrix)


class PalPA2:

	def __init__(self, dim=7, c=0.1, L=8):
		self.c=c
		self.dict_length = dim
		self.number_of_classes = L
		self.weights = self.init_weights()
		self.error_rate = 0.0
		self.correct_classified = 0.0
		self.incorrect_classified = 0.0
		self.number_of_rounds = 0.0

	def init_weights(self):
		
		weights = []
		for i in range(0,self.number_of_classes):
			weights.append([0.0] * self.dict_length)
		
		return weights

	def predict_label(self, feature_vectors):
		
		maxi = -99999999.0
		label = 0
		for i in range(0,len(self.weights)):
			total = 0.0
			for eachVector in range(0,len(feature_vectors)):
				total += feature_vectors[eachVector]*self.weights[i][eachVector]
			if total >= maxi:
				maxi = total
				label = i
		
		return label

	def update_weights(self, update_matrix):
		
		for i in range(0,len(self.weights)):
			for j in range(0,len(self.weights[i])):
				self.weights[i][j] += update_matrix[i][j]


	def get_update_matrix(self, feature_vectors, predicted_label, label_set, support_set2,l_tilde,labeltil):
		update_matrix = self.init_weights()
		step_lambda2 = [0.0 for x in range(0,len(update_matrix))] ## Lambdas for each label
		
		total_support_class_loss2 = 0.0
		
		modxsq = 0.0
		sum_step_lambda2 = 0.0
		
		for i in range(0,len(support_set2)):
			total_support_class_loss2 += l_tilde[support_set2[i]]

		### Calculating ||X||^2
		for j in range(0,len(feature_vectors)):
			modxsq += feature_vectors[j]*feature_vectors[j]

		### Determining lambdas for each support class label, non support class has lambda = 0
		for r in range(0,len(support_set2)):
			temp1 = (1/(2*self.c)) + (modxsq/len(true_label))
			temp2 = ((len(support_set2)/(2*self.c)) + ((len(support_set2)*modxsq)/len(label_set)) + modxsq)
			step_lambda2[support_set2[r]] = (l_tilde[support_set2[r]] - ((temp1/temp2)*total_support_class_loss2))/modxsq
		
		### Calculating the sum of all lambdas
		for j in range(0,len(step_lambda2)):
			sum_step_lambda2 += step_lambda2[j]

		### Determining the weight update corresponding to each label
		for i in range(0,len(update_matrix)):
			if i in true_label:
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = feature_vectors[j] * (sum_step_lambda2/len(label_set))
			else:                
				for j in range(0,len(feature_vectors)):
					update_matrix[i][j] = -feature_vectors[j] * (step_lambda2[i])

		return update_matrix
		

	def run(self, feature_vectors, true_label, label_set):
		self.number_of_rounds += 1.0
		predicted_label = self.predict_label(feature_vectors)
		
		if predicted_label==true_label:
			self.correct_classified += 1.0
		else:
			self.incorrect_classified += 1.0
		self.error_rate = self.incorrect_classified/self.number_of_rounds
		
		
		### Calculating l_tilde for each label
		l_tilde = [-1.0 for x in range(len(self.weights))]
		lt_tilde = [-1.0 for x in range(len(self.weights))]
		labeltil = [-1 for x in range(len(self.weights))]
		label_set_total = 0.0
		
		for r in range(0,len(label_set)):
			for eachVector in range(0,len(feature_vectors)):
				label_set_total +=  feature_vectors[eachVector]*self.weights[r][eachVector]
		
		label_set_total = label_set_total/len(label_set)
		modx=0.0
		j = 0
		
		for j in range(0,len(feature_vectors)):
			modx += feature_vectors[j]*feature_vectors[j]        
		for r in range(0,len(self.weights)):
			if r not in label_set:
				loss_r = 0.0

				for eachVector in range(0,len(feature_vectors)):
					loss_r += feature_vectors[eachVector]*self.weights[r][eachVector]
				l_tilde[r] = max(0.0, 1 - label_set_total + loss_r)
				lt_tilde[r] = max(0.0, 1 - label_set_total + loss_r)
				labeltil[r] = r


		### Sorting in Decreasing order the l_tilde copy and corresponding label
		for i in range(len(self.weights)):
			for j in range(0, len(self.weights)-i-1):
				if lt_tilde[j] < lt_tilde[j+1]:
					lt_tilde[j], lt_tilde[j+1] = lt_tilde[j+1], lt_tilde[j]
					labeltil[j], labeltil[j+1] = labeltil[j+1], labeltil[j]


		support_set2 = []
		
		cm_loss2 = 0.0 # cumulative loss
		

		### Determining Support Classes
		mods = len(l_tilde)
		if(lt_tilde[0] != 0.0):
			support_set2.append(labeltil[0])
			j = 1
			cm_loss2 += l_tilde[labeltil[j-1]]

			while(j!= mods):
				if (labeltil[j] != -1 and l_tilde[labeltil[j]] != -1.0):
					temp1 = (j/(2*self.c)) + (modx*((j/len(label_set)) + 1))
					temp2 = (1/(2*self.c)) + (modx/len(label_set))
					rhs = ((temp1/temp2)*l_tilde[labeltil[j]])

					if(cm_loss2 < rhs):
						support_set2.append(labeltil[j])
						cm_loss2 += l_tilde[labeltil[j]]
						j += 1
					
					else:
						break
				else:
					break

		# Updating weight matrix using the determined support class set and losses
		update_matrix = self.get_update_matrix(feature_vectors, predicted_label, label_set, support_set2,l_tilde,labeltil)
		self.update_weights(update_matrix)