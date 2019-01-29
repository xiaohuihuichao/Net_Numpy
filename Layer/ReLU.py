import numpy as np

class ReLULayer(object):
    def __init__(self, dim):
        self.__type = "activation"
        self.__name = "relu"
        self.__dim = dim

    def type(self):
        return self.__type
    def name(self):
        return self.__name

    def forward(self, input):
        self.__input = input
        return np.maximum(self.__input, 0)

    def backward(self, delta):
        delta[self.__input<0]=0
        return delta

    def getInDim(self):
        return self.__dim
    def getOutDim(self):
        return self.__dim

class LeakyReLULayer(object):
    def __init__(self, dim, alpha=0.1):
        self.__type = "activation"
        self.__name = "leaky_relu"
        self.__dim = dim
        self.__alpha = alpha

    def type(self):
        return self.__type
    def name(self):
        return self.__name

    def forward(self, input):
        self.__output = np.maximum(input, 0) + self.__alpha*np.minimum(input, 0)
        return self.__output

    def backward(self, delta=0.1):
        delta[self.__output<0] *= self.__alpha
        return delta

    def getInDim(self):
        return self.__dim
    def getOutDim(self):
        return self.__dim



if __name__ == "__main__":
    img = np.array([[-1., -2, 3, 4, 5, 6, -7, -8], [8, -7, 6, 5, 4, -3, 2, -1]])

    leakyrelu = LeakyReLULayer(8)
    print(leakyrelu.forward(img))
    print(leakyrelu.backward(np.array([[-1., -2, 3, 4, 5, 6, -7, -8], [8, -7, 6, 5, 4, -3, 2, -1]])))

'''
    relu = ReLULayer(8)
    print(relu.forward(img))
    print(relu.backward(np.array([[-1, -2, 3, 4, 5, 6, -7, -8], [8, -7, 6, 5, 4, -3, 2, -1]])))
'''