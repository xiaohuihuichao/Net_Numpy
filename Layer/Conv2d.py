import numpy as np
import tool.im2col as tool

# img_shape:[h, w, channels]
class Conv2D(object):
    def __init__(self, img_shape, \
                output_channels, \
                ksize_h=3, ksize_w=3, strides=1, mode="VALID"):
        self.__type = "vision"
        self.__name = "conv"
        self.__img_shape = img_shape
        self.__ksize_h = ksize_h
        self.__ksize_w = ksize_w
        self.__strides = strides
        self.__output_channels = output_channels
        self.__mode = mode


        self.__w = np.random.standard_normal(
            (self.__ksize_w, self.__ksize_h, self.__img_shape[-1], self.__output_channels))
        self.__b = np.random.standard_normal(self.__output_channels)

        self.__dw = np.zeros(self.__w.shape)
        self.__db = np.zeros(self.__b.shape)


    def forward(self, img):
        if self.__mode == "SAME":
            # img:[batch_size, w, h, channels]
            img = np.pad(img,
            ((0, 0),
            (self.__ksize_w//2, self.__ksize_w//2),
            (self.__ksize_h//2, self.__ksize_h//2),
            (0, 0)),
            "constant", constant_values=0.)
            # conv_out:[batch_size, w, h, channels]
            conv_out = np.zeros((img.shape[0],
                self.__img_shape[0]//self.__strides,
                self.__img_shape[1]//self.__strides,
                self.__output_channels))
        elif self.__mode == "VALID":
            conv_out = np.zeros((img.shape[0],
                (self.__img_shape[0]-self.__ksize_w+1)//self.__strides,
                (self.__img_shape[1]-self.__ksize_h+1)//self.__strides,
                self.__output_channels))
        else:
            print("Conv mode is wrong.\n")

        self.__batch_size = img.shape[0]

        self.__img_col = []
        # [kw*kh*in_channel, out_channel]
        w_col = self.__w.reshape([-1, self.__output_channels])
        for i in range(self.__batch_size):
            # [w, h, in_channel]
            img_i = img[i]
            # [w*h, kw*kh*in_channel]
            img_col_i = tool.im2col(img_i,
                    self.__ksize_w, self.__ksize_h, self.__strides)
            # [w*h, out_channel]
            col_out = np.dot(img_col_i, w_col) + self.__b
            conv_out[i] = np.reshape(col_out, conv_out.shape[1:])
            self.__img_col.append(img_col_i)
        # [batch_size, w*h, kw*kh*in_channel]
        self.__img_col = np.array(self.__img_col)
        # [batch_size, w, h, out_channel]
        return conv_out

    def _gradient(self, eta):
        # [batch_size, w*h, out_channel]
        self.__dw = np.zeros(self.__w.shape)
        self.__db = np.zeros(self.__b.shape)

        eta_col = np.reshape(eta, [self.__batch_size, -1, self.__output_channels])
        for i in range(self.__batch_size):
            self.__dw += np.dot(self.__img_col[i].T, eta_col[i]).reshape(self.__w.shape)
        self.__db = np.sum(eta_col, axis=(0, 1)).reshape(self.__b.shape)
        self.__db /= (self.__batch_size * eta_col.shape[1])#shape[1]:w*h
        self.__dw /= (self.__batch_size * eta_col.shape[1])

        # img_col_i * w_col
        if self.__mode == "VALID":
            eta_pad = np.pad(eta,
                ((0, 0),
                (self.__ksize_w - 1, self.__ksize_w - 1),
                (self.__ksize_h - 1, self.__ksize_h - 1),
                (0, 0)),
                'constant', constant_values=0)
        elif self.__mode == "SAME":
            eta_pad = np.pad(eta,
                ((0, 0),
                (self.__ksize_w // 2, self.__ksize_w // 2),
                (self.__ksize_h // 2, self.__ksize_h // 2),
                (0, 0)),
                'constant', constant_values=0)
        else:
            print("Conv2D mode is wrong.\n")

        flip_w = (np.fliplr(np.flipud(self.__w))).swapaxes(2, 3)
        # [kw*kh*out_channel, in_channel]
        flip_w_col = flip_w.reshape([-1, self.__img_shape[-1]])
        eta_pad_col = np.array([tool.im2col(eta_pad[i], self.__ksize_w, self.__ksize_h, self.__strides) for i in range(self.__batch_size)])

        next_eta = np.dot(eta_pad_col, flip_w_col)
        
        # [batch_size, w, h, in_channel]
        input_shape = (self.__batch_size, self.__img_shape[0], 
                        self.__img_shape[1], self.__img_shape[2])
        next_eta = np.reshape(next_eta, input_shape)
        return next_eta


    def backward(self, eta, alpha=1e-4, weight_decay=4e-4):
        next_eta = self._gradient(eta)

        self.__w *= (1 - weight_decay)
        self.__b *= (1 - weight_decay)
        

        self.__w -= alpha * self.__dw
        self.__b -= alpha * self.__db

        return next_eta

    def getW(self):
        return self.__w.copy()
    def getB(self):
        return self.__b.copy()
    def getDw(self):
        return self.__dw.copy()
    def getDb(self):
        return self.__db.copy()




if __name__ == "__main__":
    import matplotlib.pyplot as plt # plt 用于显示图片
    import matplotlib.image as mpimg # mpimg 用于读取图片
    import numpy as np
    import time

    lena = mpimg.imread('data/lena.png').astype(np.float)
    if len(lena.shape) == 2:
        lena = np.dstack((lena,lena,lena))
    elif lena.shape[2] == 4:
        lena = lena[:, :, :3]

    img_h = lena.shape[0]
    img_w = lena.shape[1]
    plt.figure(0)
    plt.imshow(lena)
    plt.savefig('data/lena_.png')

    lena = np.reshape(lena, [1, img_h, img_w, 3])
    print(lena.shape)
    conv = Conv2D(lena.shape[1:], \
        output_channels=3, ksize_h=3, ksize_w=5, strides=1, \
        mode="SAME")

    aa = conv.forward(lena)
    plt.figure(1)
    plt.imshow(aa.reshape([img_h, img_w, 3]))
    #plt.show()
    plt.savefig("data/fig_0.png")

    ite = 20000
    start = time.clock()
    for i in range(1, ite+1):
        #plt.close()
        next = conv.forward(lena)
        eta = conv.backward(next-lena, alpha=5e-2, \
                        weight_decay=1e-4)
        if i % (ite//100) == 0:
            print(str(i) + " next-lena:")
            print("  " + str(np.abs((next-lena)).sum()))
            plt.figure(2)
            next = next.reshape([img_h, img_w, 3])
            plt.imshow(next)
            plt.savefig('data/fig_%d.png' % i)

    plt.figure(2)
    plt.imshow(lena.reshape([img_h, img_w, 3]))
    plt.figure(3)
    plt.imshow(next.reshape([img_h, img_w, 3]))
    plt.axis('off')
    plt.savefig('data/fig_out.png')
    #plt.show()

    print("next-lena:")
    print("  " + str(np.abs((next-lena)).sum()))
    print(str((time.clock()-start)/ite) + "s")

    '''
    # img = np.random.standard_normal((2, 32, 32, 3))
    import time
    batch = 10
    in_channels = 3
    out_channels = 8
    img_h = 32
    img_w = 32
    ite = 1000

    img = np.ones((batch, img_h, img_w, in_channels))
    # print(img[0].dtype) #float64
    conv = Conv2D(img.shape[1:],
        output_channels=out_channels, ksize_h=3, ksize_w=5, strides=1,
        mode="SAME")


    next1 = np.ones((batch, img_h, img_w, out_channels))
    next = conv.forward(img)
    print("next1-next")
    print("  " + str(np.abs((next-next1)).sum()))


    start = time.clock()
    for i in range(1, ite+1):
        next = conv.forward(img)
        #print(next.shape)
        #print("next")
        #print(next.shape, next1.shape)
        eta = conv.backward(next-next1, alpha=1e-4, weight_decay=4e-4)
        if i % (ite//10) == 0:
            print(str(i) + "next1-next:")
            print("  " + str(np.abs((next-next1)).sum()))
        #print("eta")
        ##print(eta)
        #print("next1-next")
        #print("\t" + str(np.abs((next-next1)).sum()))
        #print("dw")
        #print(conv.getDw())
        #print("db")
        #print(conv.getDb())
    print("next1-next")
    print("  " + str(np.abs((next-next1)).sum()))
    print(str((time.clock()-start)/ite) + "s")
    '''