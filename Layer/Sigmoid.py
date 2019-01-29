import numpy as np

class SigmoidLayer(object):
    def __init__(self, dim):
        self.__type = "activation"
        self.__name = "sigmoid"
        self.__dim = dim

    def type(self):
        return self.__type
    def name(self):
        return self.__name

    def forward(self, input):
        self.__output = 1./(1.+np.exp(-input))
        return self.__output

    def backward(self, delta):
        return self.__output * (1 - self.__output) * delta

    def getInDim(self):
        return self.__dim
    def getOutDim(self):
        return self.__dim

if __name__ == "__main__":
    img = np.array([[-1, -2, 3, 4, 5, 6, -7, -8], [8, -7, 6, 5, 4, -3, 2, -1]])
    sigmoid = SigmoidLayer(8)

    print(sigmoid.forward(img))
    print(sigmoid.backward(np.array([[-1, -2, 3, 4, 5, 6, -7, -8], [8, -7, 6, 5, 4, -3, 2, -1]])))