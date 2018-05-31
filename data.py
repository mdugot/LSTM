import numpy as np
import random
from tqdm import tqdm

numSteps = 1440
numFeatures = 11

class Data:

	def __init__(self):
		print("load train data")
		self.o_traindata = self.readData("train.csv")
		print("load test data")
		self.o_testdata = self.readData("test.csv")
		print("load output data")
		self.o_output = self.readData("output.csv", ";")
		print("normalize")
		self.normalize()
		print("split training set")
		self.n_validationdata = self.n_traindata[11000:]
		self.n_traindata = self.n_traindata[:11000]
		print("convert to dictionary")
		self.traindata = self.convertToDict(self.n_traindata)
		self.validationdata = self.convertToDict(self.n_validationdata)
		self.testdata = self.convertToDict(self.n_testdata)
		self.output = self.convertToDict(self.o_output)

		self.trainKeys = np.array(self.traindata.keys())

	def readData(self, filename, delimiter = ","):
		with open(filename) as file:
			names = None
			lines = file.readlines()
			names = lines[0].strip().split(delimiter)
			data = np.zeros([len(lines)-1, len(names)], dtype=np.float64)
			
			for i in tqdm(range(len(lines))):
				if i > 0 :
					line = lines[i].strip()
					line = np.fromstring(line, sep=delimiter)
					data[i - 1] = line
		return data
	
	def normalize(self):
		print("  remove id and reshape temporaly")
		train_tmp = np.reshape(self.o_traindata[:,1:], [-1, numFeatures])
		test_tmp = np.reshape(self.o_testdata[:,1:], [-1, numFeatures])

		print("  get stats")
		mins = np.min(train_tmp, 0)
		maxs = np.max(train_tmp, 0)
		means = np.mean(train_tmp, 0)
		ranges = maxs - mins
		ranges[ranges == 0] = 1
		print("  normalize data")
		train_tmp = (train_tmp - means) / ranges
		test_tmp = (test_tmp - means) / ranges
		print("  reshape to origin")
		train_tmp = np.reshape(train_tmp, [-1, numFeatures * numSteps])
		test_tmp = np.reshape(test_tmp, [-1, numFeatures * numSteps])
		print("  reset id")
		self.n_traindata = np.c_[self.o_traindata[:,0], train_tmp]
		self.n_testdata = np.c_[self.o_testdata[:,0], test_tmp]
	
	def convertToDict(self, data):
		result = dict()
		for row in tqdm(data):
			result[int(row[0])] = row[1:]
		return result
	
	def steps(self, data, size = 500):
		selections = np.sort(np.random.choice(numSteps, size, replace = False))
		return np.reshape(data, [numSteps, numFeatures])[selections]
		#return np.reshape(data, [numSteps, numFeatures])[-size:]
		#return np.reshape(data, [numSteps, numFeatures])[:size]
	
	def randomBatch(self, batchSize, stepsLength = 500):
		result = np.zeros([batchSize, stepsLength, numFeatures])
		outputs = np.zeros([batchSize, 1])
		sampleKeys = random.sample(self.traindata.keys(), batchSize)
		#print(str(sampleKeys))
		for i in range(len(sampleKeys)):
			result[i] = self.steps(self.traindata[sampleKeys[i]], stepsLength)
			outputs[i] = self.output[sampleKeys[i]]
		return result,outputs
	
	def validationBatch(self, start, batchSize, stepsLength = 500):
		sampleKeys = list(self.validationdata.keys())[start:start+batchSize]
		result = np.zeros([len(sampleKeys), stepsLength, numFeatures])
		outputs = np.zeros([len(sampleKeys), 1])
		#print(str(sampleKeys))
		for i in range(len(sampleKeys)):
			result[i] = self.steps(self.validationdata[sampleKeys[i]], stepsLength)
			outputs[i] = self.output[sampleKeys[i]]
		return result,outputs

	def save(self, nn, filename, limit = 0.5, numberTry = 11):
		size = len(self.testdata)
		result = np.zeros([size, 2], dtype=np.int32)
		i = 0
		for k in tqdm(list(self.testdata.keys())):
			p = nn.predict(self, self.testdata[k], limit, numberTry)
			result[i][0] = k
			result[i][1] = p
			i += 1
		np.savetxt(filename, result, "%s", delimiter=";", header = "ID;label", comments="")
