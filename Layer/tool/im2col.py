import numpy as np

# img:[width ,height, in_channel]
# weight * im2col
# result is a row vector
def im2col(img, ksize_w=3, ksize_h=3, strides=1):
    img_col = []
    for i in range(0, img.shape[0] - ksize_w + 1, strides):
        for j in range(0, img.shape[1] - ksize_h + 1, strides):
            col = img[i:i + ksize_w, j:j + ksize_h, :].reshape([-1])
            img_col.append(col)
    img_col = np.array(img_col)

    return img_col

if __name__ == "__main__":
    img = np.array(range(1, 26))
    img = img.reshape((5, 5, -1))
    print(im2col(img, 5, 3, 1))