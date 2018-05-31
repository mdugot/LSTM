import numpy as np
import tensorflow as tf
import random
import data
from tqdm import tqdm

class NeuralNetwork:

	def standardCell(self, lastOutput, state, inputs):
		concatSize = self.stateSize + self.features

		with tf.variable_scope("standard", reuse=tf.AUTO_REUSE):

			tanhGate_w = tf.get_variable(
				name = "tanh_w",
				shape = [concatSize, self.stateSize],
				dtype = tf.float64,
				initializer = self.winit)
			tanhGate_b = tf.get_variable(
				name = "tanh_b",
				shape = [self.stateSize],
				dtype = tf.float64,
				initializer = self.binit)

		concatInputs = tf.concat([inputs, state], axis=1)
		tanhOutput = tf.tanh(tf.matmul(concatInputs, tanhGate_w) + tanhGate_b)
		return tanhOutput, tanhOutput

	def LSTM(self, lastOutput, state, inputs):
		concatSize = self.stateSize + self.features

		with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):

			forgetGate_w = tf.get_variable(
				name = "forget_w",
				shape = [concatSize, self.stateSize],
				dtype = tf.float64,
				initializer = self.winit)
			forgetGate_b = tf.get_variable(
				name = "forget_b",
				shape = [self.stateSize],
				dtype = tf.float64,
				initializer = self.binit)

			inputGate_w = tf.get_variable(
				name = "input_w",
				shape = [concatSize, self.stateSize],
				dtype = tf.float64,
				initializer = self.winit)
			inputGate_b = tf.get_variable(
				name = "input_b",
				shape = [self.stateSize],
				dtype = tf.float64,
				initializer = self.binit)

			tanhGate_w = tf.get_variable(
				name = "tanh_w",
				shape = [concatSize, self.stateSize],
				dtype = tf.float64,
				initializer = self.winit)
			tanhGate_b = tf.get_variable(
				name = "tanh_b",
				shape = [self.stateSize],
				dtype = tf.float64,
				initializer = self.binit)

			sigmoidGate_w = tf.get_variable(
				name = "sigmoid_w",
				shape = [concatSize, self.stateSize],
				dtype = tf.float64,
				initializer = self.winit)
			sigmoidGate_b = tf.get_variable(
				name = "sigmoid_b",
				shape = [self.stateSize],
				dtype = tf.float64,
				initializer = self.binit)
		self.forgetGate_w = forgetGate_w
		self.forgetGate_b = forgetGate_b
		self.inputGate_w = inputGate_w
		self.inputGate_b = inputGate_b
		self.tanhGate_w = tanhGate_w
		self.tanhGate_b = tanhGate_b
		self.sigmoidGate_w = sigmoidGate_w
		self.sigmoidGate_b = sigmoidGate_b

		#concatInputs = tf.nn.dropout(tf.concat([inputs, lastOutput], axis=1), self.keepProb)
		concatInputs = tf.concat([inputs, lastOutput], axis=1)

		forgetOutput = tf.sigmoid(tf.matmul(concatInputs, forgetGate_w) + forgetGate_b)
		forgetState = tf.multiply(forgetOutput, state)

		inputOutput = tf.sigmoid(tf.matmul(concatInputs, inputGate_w) + inputGate_b)
		tanhOutput = tf.tanh(tf.matmul(concatInputs, tanhGate_w) + tanhGate_b)
		update = tf.multiply(inputOutput, tanhOutput)
		updateState = tf.add(forgetState, update)

		sigmoidOutput = tf.sigmoid(tf.matmul(concatInputs, sigmoidGate_w) + sigmoidGate_b)
		newOutput = tf.multiply(tf.tanh(updateState), sigmoidOutput)

		return newOutput, updateState


	def __init__(self, steps = 500, features = 11, stateSize = 50):
		tf.reset_default_graph()
		self.winit = tf.glorot_uniform_initializer()
		self.binit = tf.constant_initializer(0.0)


		self.features = features
		self.stateSize = stateSize
		self.steps = steps

		self.batchSize = tf.placeholder(tf.int32, [])
		self.inputs = tf.placeholder(tf.float64, [None, self.steps, data.numFeatures])
		self.labels = tf.placeholder(tf.float64, [None, 1])

		self.learningRate = tf.placeholder_with_default(np.float64(0.01),  [])
		self.keepProb = tf.placeholder_with_default(np.float64(0.9),  [])

		self.state = tf.fill([self.batchSize, self.stateSize], np.float64(0))
		self.firstOutput = tf.fill([self.batchSize, self.stateSize], np.float64(0)) 

		lastState = self.state
		lastOutput = self.firstOutput
		for step in tqdm(range(self.steps)):
			stepInput = tf.gather(params = self.inputs, indices = step, axis = 1)
			lastOutput, lastState = self.LSTM(lastOutput, lastState, stepInput)
			#lastOutput, lastState = self.standardCell(lastOutput, lastState, stepInput)

		self.outputs = lastOutput
		self.lastState = lastState

		with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
			outputLayer_w = tf.get_variable(
				name = "output_w",
				shape = [self.stateSize, 1],
#				shape = [self.features, 1],
				dtype = tf.float64,
				initializer = self.winit)
			outputLayer_b = tf.get_variable(
				name = "output_b",
				shape = [1],
				dtype = tf.float64,
				initializer = self.binit)

		self.predictions = tf.sigmoid(tf.matmul(self.outputs, outputLayer_w) + outputLayer_b)
		#self.predictions = tf.sigmoid(tf.matmul(stepInput, outputLayer_w) + outputLayer_b)
		self.cost = tf.reduce_mean(tf.losses.log_loss(self.labels, self.predictions))
		print("create optimizer...")
		self.optim = tf.train.AdamOptimizer(self.learningRate).minimize(self.cost)

		print("start session...")
		self.session = tf.Session()
		print("initialize variables...")
		self.session.run(tf.global_variables_initializer())

	def train(self, data, iterations, batchSize = 1000, learningRate = 0.01, keepProb = 0.9):
		cost = 0
		for i in tqdm(range(iterations)):
			batchInput, batchOutput = data.randomBatch(batchSize)
			_, c = self.session.run([self.optim, self.cost], feed_dict={
				self.batchSize: batchSize,
				self.inputs: batchInput,
				self.labels:batchOutput,
				self.learningRate:learningRate,
				self.keepProb:keepProb
			})
			cost = cost + c
		cost = cost / iterations
		print("cost : " + str(cost))

	def getSuccess(self, predictions, labels, limit = 0.5):
		predictions[predictions < limit] = 0
		predictions[predictions >= limit] = 1
		success = predictions[predictions == labels]
		return len(success)
	
	def predict(self, data, inputs, limit = 0.5, numberTry = 11):
		p = 0
		numberTry = 11
		for i in range(numberTry):
			p += self.session.run(self.predictions, feed_dict={
				self.batchSize: 1,
				self.inputs: [data.steps(inputs)],
				self.keepProb:1
			})
		p /= numberTry
		if (p >= limit):
			return 1
		else:
			return 0

	def check(self, data, batchSize = 1000, limit = 0.5):
		success = 0
		total = 0
		i = 0
		iterations = 0
		cost = 0
		while i < len(data.validationdata):
			batchInput, batchOutput = data.validationBatch(i, batchSize)
			p, c = self.session.run([self.predictions, self.cost], feed_dict={
				self.batchSize: len(batchOutput),
				self.inputs: batchInput,
				self.labels:batchOutput,
				self.keepProb:1
			})
			total = total +len(batchOutput) 
			success = success + self.getSuccess(p, batchOutput, limit)
			i = i + batchSize
			cost = cost + c
			iterations = iterations + 1
		cost = cost / iterations
		print("cost : " + str(cost))
		print("success : " + str(success) + "/" + str(total))
		print("rate : " + str(int(success / total * 100)) + "%")

	def check2(self, data, limit = 0.5):
		success = 0
		total = 0
		i = 0
		iterations = 0
		cost = 0
		for k in tqdm(list(data.validationdata.keys())):
			p = self.predict(data, data.validationdata[k])
			total = total + 1
			if p == data.output[k]:
				success = success + 1
		print("success : " + str(success) + "/" + str(total))
		print("rate : " + str(int(success / total * 100)) + "%")

