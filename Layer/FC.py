import numpy as np
import math
from functools import reduce


class FullyConnectionLayer(object):
    def __init__(self, input_len, output_len=3):
        self.__type = "vision"
        self.__name = "fc"
        self.__output_len = output_len
        self.__input_len = input_len

        self.__w = np.random.standard_normal((self.__input_len, self.__output_len)) / 10
        self.__b = np.random.standard_normal(self.__output_len) / 10

        self.__dw = np.zeros(self.__w.shape)
        self.__db = np.zeros(self.__b.shape)


    def type(self):
        return self.__type
    def name(self):
        return self.__name

    def forward(self, input):
        #print(input.shape)
        self.__input = input.reshape([input.shape[0], -1])
        return np.dot(self.__input, self.__w) + self.__b

    def backward(self, delta, alpha=0.001, lamda=0.):
        next_delta = self._gradient(delta)
        weight_decay = alpha * lamda
        self.__w *= (1 - weight_decay)
        self.__b *= (1 - weight_decay)

        self.__w -= alpha * self.__dw
        self.__b -= alpha * self.__db

        self.__dw = np.zeros(self.__w.shape)
        self.__db = np.zeros(self.__b.shape)

        return next_delta

    # delta [batchsize, output_len]
    # __input [batchsize, __input_len]
    # __w [input_len, output_len]
    # __b [1, output]
    def _gradient(self, delta):
        self.__dw = np.dot(self.__input.T, delta) / delta.shape[0]
        self.__db = np.sum(delta, axis=0)
        '''
        for i in range(delta.shape[0]):
            input_i = self.__input[i][:, np.newaxis]
            delta_i = delta[i][:, np.newaxis]
            self.__dw += np.dot(delta_i.T, input_i)
            self.__db += delta_i.reshape(self.__b.shape)
        self.__dw /= delta.shape[0]
        self.__db /= delta.shape[0]
        '''

        next_delta = np.dot(delta, self.__w.T)
        return next_delta

    def getOutDim(self):
        return self.__output_len
    def getInDim(self):
        return self.__output_len

    def getW(self):
        return self.__w
    def getB(self):
        return self.__b
    def getDW(self):
        return self.__dw
    def getDB(self):
        return self.__db



if __name__ == "__main__":
    img = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]])
    fc = FullyConnectionLayer(8, 3)
    out = fc.forward(img)

    print(fc.backward(np.array([[1, -2, 1], [3, 4, 0]])))

    print(fc.getW())
    print(fc.getB())