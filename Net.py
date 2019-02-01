import numpy as np
import Layer.FC as FC
import Layer.ReLU as ReLU
import Layer.Tanh as Tanh
import Layer.Sigmoid as Sigmoid
import Layer.Softmax as Softmax
import matplotlib.pyplot as plt


def isEqual(num1, num2):
	if num1+1e-4<=num2 and num1-1e-4>=num2:
		return True
	return False

class Net(object):
	def __init__(self, batch_size=200,
			base_lr=0.1, lr_decay_scale=1., lr_decay_num=-1,
			weight_decay=0.,
			mode="SGD", show=True):
		self.__batch_size = batch_size
		self.__lr = base_lr
		self.__lr_decay_scale = lr_decay_scale
		self.__lr_decay_num = lr_decay_num
		self.__weight_decay = weight_decay
		self.__mode = mode
		self.__network = []
		self.__show = show
		self.__ite = -1


	def clearStructure(self):
		self.__network[:] = []


	def loadData(self, data, label):
		self.__data = data
		self.__label = label
		self.__data_num = self.__data.shape[0]
		self.__data_dim = self.__data.shape[1]
		self.__label_dim = label.shape[1]
		if self.__batch_size > self.__data_num:
			self.__batch_size = self.__data_num

		#self.__batch_data = np.zeros([self.__batch_size, self.__data_dim])
		#self.__batch_label = np.zeros([self.__batch_size, self.__label_dim])

		if self.__show:
			show_message = "Loading data...\n\tInput:[%d, %d], label:[%d, %d]." % (self.__batch_size, self.__data_dim,
				self.__batch_size, self.__label_dim)
			print(show_message)


	def addFC(self, num_output):
		if len(self.__network) == 0:
			self.__network.append(FC.FullyConnectionLayer(self.__data_dim, num_output))
		else:
			last_out_dim = self.__network[-1].getOutDim()
			# input_len = reduce(lambda x, y: x * y, shape[1:])
			self.__network.append(FC.FullyConnectionLayer(last_out_dim, num_output))

		if self.__show:
			show_message = "Layer %d\n\tFully Connection Layer, input:[%d, %d], output:[%d, %d]." % (
						len(self.__network),
						self.__batch_size, self.__network[-1].getInDim(),
						self.__batch_size, self.__network[-1].getOutDim())
			print(show_message)


	def addSigmoid(self, dim):
		self.__network.append(Sigmoid.SigmoidLayer(dim))
		if self.__show == True:
			show_message = "Layer %d\n\tSigmoid Layer." % (
						len(self.__network))
			print(show_message)


	def addReLU(self, dim):
		self.__network.append(ReLU.ReLULayer(dim))
		if self.__show == True:
			show_message = "Layer %d\n\tReLU Layer." % (
						len(self.__network))
			print(show_message)


	def addLeakyReLU(self, dim, alpha=0.1):
		self.__network.append(ReLU.LeakyReLULayer(dim, alpha))
		if self.__show:
			show_message = "Layer %d\n\tLeakyReLU Layer." % (
						len(self.__network))
			print(show_message)


	def addTanh(self, dim):
		self.__network.append(Tanh.TanhLayer(dim))
		if self.__show == True:
			show_message = "Layer %d\n\tTanh Layer." % (
						len(self.__network))
			print(show_message)


	def addSoftmax(self):
		self.__network.append(Softmax.SoftmaxWithLossLayer())
		if self.__show == True:
			show_message = "Layer %d\n\tSoftmax(With Loss) Layer." % (
						len(self.__network))
			print(show_message)


	def getNetwork(self):
		return self.__network


	def setWeightDecay(self, lamda):
		self.__weight_decay = lamda


	def setLrDecay(self, lr_decay_scale, lr_decay_num):
		self.__lr_decay_scale = lr_decay_scale
		self.__lr_decay_num = lr_decay_num


	def setOptimizer(self, batch_size = 200,
			lr_decay_scale=1., lr_decay_num=-1,
			weight_decay=0., mode="SGD"):
		self.__batch_size = batch_size
		self.__lr_decay_scale = lr_decay_scale
		self.__lr_decay_num = lr_decay_num
		self.__weight_decay = weight_decay
		self.__mode = mode

	# [index_start, index_end)
	def _loadBatch(self):
		self.__ite += 1
		index_start = self.__ite * self.__batch_size
		index_end = index_start+self.__batch_size
		if index_end >= self.__data_num:
			index_end = self.__data_num
			self.__ite = -1

		self.__batch_data = self.__data[index_start:index_end, :]
		self.__batch_label = self.__label[index_start:index_end, :]
		if self.__network[-1].type() == "loss":
			self.__network[-1].loadLabel(self.__batch_label)
		else:
			print("The last layer is not loss.")


	def forward(self):
		if len(self.__network) == 0:
			print("The network is empty.\n")
		else:
			self._loadBatch()
			data_flow = self.__batch_data

			for index in range(len(self.__network)):
				if index==(len(self.__network)-1) and self.__network[index].type()!="loss":
						print("The last layer is not loss layer.")
				data_flow = self.__network[index].forward(data_flow)


	def backward(self):
		#for index in reversed(range(len(self.__network)))
		if self.__network[-1].type()=="loss":
			delta = self.__network[-1].backward()
		else:
			print("The last layer is not loss layer.")

		for index in range(len(self.__network)-2, -1, -1):
			if self.__network[index].type()=="vision":
				delta = self.__network[index].backward(delta,
					alpha=self.__lr, lamda=self.__weight_decay)
			else:
				delta = self.__network[index].backward(delta)


	def predict(self, data):
		if len(self.__network) == 0:
			print("The network is empty.\n")
		else:
			pre_result = data

			for index in range(len(self.__network)-1):
				pre_result = self.__network[index].forward(pre_result)
			#pre_result = np.argmax(data_flow, axis=1)
			return pre_result


	def accuracy(self, test_data, test_label):
		if test_data.shape[0] != test_label.shape[0]:
			print("The row of data and label is not equal.")

		pre_result = np.argmax(self.predict(test_data), axis=1)
		test_result = np.argmax(test_label, axis=1)

		num = 0.
		for i in range(len(test_result)):
			if pre_result[i] == test_result[i]:
				num += 1
		return num / len(test_result)


	def getLoss(self):
		return self.__network[-1].getLoss()


	def train(self):
		pass


	def save_structure(self, file_path="./", file_suff=".struct"):
		pass


	def save_parameters(self, file_path="./", file_suff=".param"):
		pass



if __name__ == "__main__":
	num_ = 100

	x1 = np.linspace(-10, -1e-5, num=num_//2)
	x2 = np.linspace(1e-5, 10, num=num_//2)
	x = np.array([x1, x2]).reshape(num_, -1)

	y1 = [1., 0.]*(num_//2)
	y2 = [0., 1.]*(num_//2)
	y = np.array([y1, y2]).reshape(num_, -1)

	net = Net(base_lr=0.1, lr_decay_scale=0.9, lr_decay_num=-1,
			weight_decay=0., batch_size=1000000,
			mode="SGD", show=True)
	net.clearStructure()
	net.loadData(x, y)

	net.addFC(y.shape[1])
	net.addLeakyReLU(y.shape[1])
	net.addSoftmax()
	ITE = 2000 + 1		#2000
	for ite in range(1, ITE):
		net.forward()
		net.backward()
		if ite % 200==0:
			print("%d:\n  loss:%.10f" % (ite, net.getLoss()))
			print("  accuracy:%.2f%%" % (net.accuracy(x, y)*100))
	#print(net.getNetwork()[-3].getW())
	#print(net.getNetwork()[-3].getB())
	#print(np.exp(net.predict(x)))