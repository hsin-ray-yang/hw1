import pandas as pd
import numpy as np
import sys
import math
def preprocess(df, feature):
    """Dataframe preprocess
    Preprocess the dataframe & extract specific value.

    Args:
        df: PM2.5 Dataframe
        feature: SO2, PM2.5, and so on

    Returns:
        processed dataframe

    """
    df = df[feature]
    df = df.str.extract('([0-9]+\.[0-9]+|[0-9]+)').astype('float32')
    df.fillna(value=0, inplace=True)
    df = np.array(df).flatten().reshape(1, -1)
    return df

def validation(inputData, outputLabel):
    """Data Cleansing
    
    Args:
        inputData: train_data
        outputLabel: train_label

    """
    if outputLabel < 0 or outputLabel > 65:
        return False
    for i in range(9):
        if inputData[i,10] < 0 or inputData[i,10] > 65:
            return False
    return True


class Model:
	def __init__(self):
		self.w = np.zeros((1,136))
		# SO2,NO,NOx,NO2,CO,O3,THC,CH4,NMHC,PM10,PM2.5,WS,WD,AT,RH 
		pre = np.array([[0.3,0.3,0.3,0.3,0.3,0.3,0,0,0.3,1,1,0,0,0,0]])
		self.preception = pre
		for i in range(8):
			pre = pre*1.1
			pre[:,9] = pre[:,9]+1/(1+math.exp(-i+9))
			pre[:,10] = pre[:,10]+1.5/(1+math.exp(-i+9))
			self.preception = np.concatenate((self.preception,pre),axis = 1)
		self.preception = self.preception/self.preception.mean()
		self.preception = np.concatenate((self.preception,np.ones((self.preception.shape[0],1))),axis =1)
		print(self.preception)
	
	def run(self,data):
		return np.sum(np.multiply(self.w,data),axis=1).reshape((-1,1))

	def gradient_decent(self,gradient,learning_rate):
		gradient = gradient.reshape((1,-1))
		# print(gradient.shape)
		self.w = self.w - learning_rate * np.multiply(gradient,self.preception)

	def save(self,dir):
		np.save(dir+'weights.npy', self.w)
		np.save(dir+'perceptions.npy', self.preception)

	def load(self,dir):
		self.w = np.load(dir+'weights.npy')
		self.preception = np.load(dir+'perceptions.npy')



class Linear_regression:
	def __init__(self,feature_scaling=False):
		self.model = Model()
		self.batch_size =1
		self.lr = 1e-3
		self.feature_scaling = feature_scaling
		self.sum_grad = np.zeros((1,136))

	def read_training_data(self,path,response=False):
		# pd.set_option("display.max_columns", None)
		# pd.set_option("display.max_rows", None)
		df = pd.read_csv(path)
		features = df.columns.values.tolist()

		isInit = False
		for feature in features:
		    series = preprocess(df, feature)
		    if isInit is False:
		        isInit = True
		        train_dataset = series
		    else:
		        train_dataset = np.concatenate((train_dataset, series), axis=0)
		        
		train_dataset = train_dataset.transpose()
		nums = train_dataset.shape[0] - 9

		isInit = False
		for hr in range(nums):
		    isValid = validation(train_dataset[hr:hr+9,:], train_dataset[hr+9, 10:11])
		    if isValid == True:
		        if isInit == False:
		            train_datas = train_dataset[hr:hr+9,:].flatten()
		            data_size = train_datas.shape[0]
		            train_labels = train_dataset[hr+9, 10:11]
		            isInit = True
		        else:
		            train_datas = np.concatenate((train_datas, train_dataset[hr:hr+9,:].flatten()))
		            train_labels = np.concatenate((train_labels, train_dataset[hr+9, 10:11]))
		            
		self.train_data = train_datas.reshape(-1, data_size)
		self.train_label = train_labels

		
		if response:
			print('------\n<<< from linear_regression: prepare_training_data\n> convert raw data to\n> self.train_data: '+str(self.train_data.shape)+'\n> self.train_label: '+str(self.train_label.shape))


	def read_testing_data(self,path,response=False):
		df = pd.read_csv(path)
		df = df.apply(pd.to_numeric, errors='coerce')
		df = df.fillna(method='ffill')
		# print(df.isnull().any())
		raw_data = df.to_numpy()
		if response:
			print('------\n<<< from linear_regression: read_testing_data\n> read testing data from: \"' +path+'\" ....\n> testing data shape= '+str(raw_data.shape))

		test_data = raw_data[::9,:]
		for i in range(1,9):
			test_data = np.concatenate((test_data,raw_data[i::9,:]),axis =1)

		if hasattr(self, 'test_data'):
			self.test_data = np.concatenate((self.test_data,test_data),axis=0)
		else:
			self.test_data = test_data

		if response:
			print('------\n<<< from linear_regression: prepare_testing_data\n> convert raw data to\n> self.test_data: '+str(self.test_data.shape)+'\n')


	def prepare_training_data(self,response=False):
		if self.feature_scaling:
			self.mean = np.mean(self.train_data,axis=0).reshape((1,-1))
			self.std = np.std(self.train_data,axis=0).reshape((1,-1))

			self.label_mean = np.mean(self.train_label,axis=0).reshape((1,-1))
			self.label_std = np.std(self.train_label,axis=0).reshape((1,-1))

			self.train_data = (self.train_data-self.mean)/self.std
			self.train_label = (self.train_label-self.label_mean)/self.label_std
			# print(self.std)



		self.train_data = np.concatenate((self.train_data,np.ones((self.train_data.shape[0],1))),axis =1)
		# np.savetxt("foo.csv", self.train_data, delimiter=",")



	def prepare_testing_data(self,response=False):
		if self.feature_scaling:
			self.test_data = (self.test_data-self.mean)/self.std

		self.test_data = np.concatenate((self.test_data,np.ones((self.test_data.shape[0],1))),axis =1)
		
		

	def run(self,data):
		return self.model.run(data)

	def loss(self,data,label):
		result = self.run(data)
		# print("------\n<<< from linear_regression: loss\n> result shape: "+str(result.shape))
		label = label.reshape((-1,1))
		loss = (label-result)**2
		return loss

	def ave_loss(self,data,label):
		if self.feature_scaling:
			loss = self.loss(data,label)
			return (np.mean(loss))*self.label_std**2
		else:
			loss = self.loss(data,label)
			return (np.mean(loss))

	def gradient(self,data,label):
		result = self.run(data)
		label = label.reshape((-1,1))
		ave_loss = self.ave_loss(data,label)
		grad = 2*np.mean(np.multiply(data , (result-label)),axis=0)
		# print(grad.shape)
		self.sum_grad += grad**2
		ada = np.sqrt(self.sum_grad)
		lr = self.lr
		self.model.gradient_decent((grad/ada).T,lr)
		return ave_loss

	def epoch(self,epoch_num = None):
		# self.gradient(self.train_data,self.train_label)
		prepared_data,prepared_label = self.prepare_batch(self.train_data,self.train_label)
		for i in range(prepared_data.shape[0]):
			loss = self.gradient(prepared_data[i,:,:],prepared_label[i,:,:])

			percentage = i/prepared_data.shape[0]*100
			sys.stdout.write('\r')
			if epoch_num is not None:
				sys.stdout.write("ep[%3d] "%(epoch_num))
			sys.stdout.write("[%-20s] %d%%" % ('='*int(percentage/5), percentage))
			sys.stdout.flush()
			print
		print(', loss= %13.2f'%(self.ave_loss(self.train_data,self.train_label)))
		# print(self.ave_loss(self.train_data,self.train_label))

	def prepare_batch(self,data,label):
		# data is 2darray #data*#feature
		label = label.reshape((-1,1))
		assert (label.shape[0] == data.shape[0])

		if self.batch_size<0:
			return (data[np.newaxis,:,:],label[np.newaxis,:,:])
		else:
			batch_size = self.batch_size
		multi = data.shape[0]//batch_size
		
		index = np.random.permutation(multi*batch_size)
		# print(index)
		data = data[index,:]
		data = data.reshape((multi,batch_size,-1))
		label = label[index,:]
		label = label.reshape((multi,batch_size,-1))
		return (data,label)

	def write_predict_result(self,data,path):
		# data has feature scaling
		result = self.run(data)
		if self.feature_scaling:
			result = (result*self.label_std)+self.label_mean
		result = result.reshape((-1))
		result[result < 0] = 0
		result = result.tolist()
		self.write_list(result,path)

	def write_list(self,data,path):
		id_name = ['id_'+str(i) for i in range(len(data))]
		df = pd.DataFrame({'id':id_name,'value':data})
		df.to_csv(path,index = False)

	def save(self,dir):
		np.save(dir+'label_std.npy',self.label_std)
		np.save(dir+'label_mean.npy',self.label_mean)
		np.save(dir+'std.npy',self.std)
		np.save(dir+'mean.npy',self.mean)
		self.model.save(dir)

	def load(self,dir):
		self.label_std = np.load(dir+'label_std.npy')
		self.label_mean = np.load(dir+'label_mean.npy')
		self.std = np.load(dir+'std.npy')
		self.mean = np.load(dir+'mean.npy')
		self.model.load(dir)





if __name__ == '__main__':
	model = Linear_regression(feature_scaling=True)
	model.read_training_data('../data/train_datas_0.csv',response=True)
	model.read_training_data('../data/train_datas_1.csv',response=True)
	model.read_testing_data(sys.argv[1],response=True)
	model.prepare_training_data(response=True)
	model.prepare_testing_data(response=True)
	print(model.loss(model.train_data,model.train_label))
	print(model.ave_loss(model.train_data,model.train_label))
	# model.gradient(model.train_data,model.train_label)
	# model.prepare_batch(model.train_data)

	for i in range(500):
		model.epoch(i)

	print(model.ave_loss(model.train_data,model.train_label))
	# print(model.ave_loss(model.test_data,model.test_label))
	model.write_predict_result(model.test_data,sys.argv[2])

	model.save('./')
	# model.write_list(model.train_data.reshape(-1).tolist(),'./label.csv')
	# model