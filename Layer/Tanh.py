import numpy as np

class TanhLayer(object):
    def __init__(self, dim):
        self.__type = "activation"
        self.__name = "tanh"
        self.__dim = dim

    def type(self):
        return self.__type
    def name(self):
        return self.__name

    def forward(self, input):
        exp_input = np.exp(input)
        exp_input_ = np.exp(-input)
        self.__output = (exp_input - exp_input_) / (exp_input + exp_input_)
        return self.__output

    def backward(self, delta):
        return delta*(1 - self.__output**2)

    def getInDim(self):
        return self.__dim
    def getOutDim(self):
        return self.__dim

if __name__ == "__main__":
    img = np.array([[-1, -2, 3, 4, 5, 6, -7, -8], [8, -7, 6, 5, 4, -3, 2, -1]])
    tanh = TanhLayer(8)

    print(tanh.forward(img))
    print(tanh.backward(np.array([[-1, -2, 3, 4, 5, 6, -7, -8], [8, -7, 6, 5, 4, -3, 2, -1]])))