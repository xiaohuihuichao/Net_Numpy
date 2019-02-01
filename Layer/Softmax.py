import numpy as np

class SoftmaxWithLossLayer(object):
    def __init__(self):
        self.__type = "loss"
        self.__name = "softmax_with_loss"

    def type(self):
        return self.__type
    def name(self):
        return self.__name

    def loadLabel(self, label):
        self.__label = label

    def forward(self, input):
        shiftx = input - np.max(input)
        exps = np.exp(shiftx)
        self.__output = exps / np.sum(exps, axis=1)[:, np.newaxis]
        #print(self.__output)

        return self.__output

    def backward(self):
        if self.__label.shape == self.__output.shape:
            self._calcLoss()
            return self.__output - self.__label
        else:
            print("Error in Softmax: self.__label.shape != input.shape.")
            print("label shape:[%d, %d], input shape:[%d, %d].\n\n" %
                            self.__label.shape[0], self.__label.shape[1],
                            self.__output.shape[0], self.__output.shape[1])

    def _calcLoss(self):
        self.__loss = -(self.__label *
                        np.log(self.__output+1e-20)).sum()/self.__label.shape[0]

    def getLoss(self):
        return self.__loss


    def getInDim(self):
        return self.__output.shape[1]
    def getOutDim(self):
        return self.__output.shape[1]


if __name__ == "__main__":
    input = np.array([[0., 0.9, 0., 0.0], [0.99, 0.01, 0.2, 0.4]])
    label = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])

    s = SoftmaxWithLossLayer()
    s.loadLabel(label)
    print(s.forward(input))

    print(s.backward())

    print(s.getLoss())