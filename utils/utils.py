import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt


def generate_input(model_name="", size=(1, 100)):
    model_name = model_name
    noise_input = np.random.normal(loc=0.0, scale=0.25, size=size)
    np.save("../raw_data/" + model_name + "/input/Input_Data.npy", noise_input)


def pad_3d_data(d_array, l=0, r=0, u=0, d=0, f=0, b=0):
    """ pad 3d data

    Parameters
    ----------
    d_array : numpy array
        data need to pad 0

    f : integer
        pad z axis forward size

    b : integer
        pad z axis backward size

    l : integer
        pad x axis left size

    r : integer
        pad x axis right size

    u : integer
        pad y axis up size

    d : integer
        pad y axis down size

    Returns
    -------
    _ : numpy array
        data after padding
    """

    n_pad = ((l, r), (u, d), (f, b))
    return np.pad(d_array, n_pad, 'constant', constant_values=0)


def pad_4d_data(in_array, a=0, c=0, f=0, b=0, l=0, r=0, u=0, d=0):
    """ pad 3d data

    Parameters
    ----------
    in_array : numpy array
        data need to pad 0

    a : integer
        pad n axis begin size

    c : integer
        pad n axis end size

    f : integer
        pad c axis forward size

    b : integer
        pad c axis backward size

    l : integer
        pad w axis left size

    r : integer
        pad w axis right size

    u : integer
        pad h axis up size

    d : integer
        pad h axis down size

    Returns
    -------
    _ : numpy array
        data after padding
    """
    n_pad = ((a, c), (f, b), (u, d), (l, r))
    return np.pad(in_array, n_pad, 'constant', constant_values=0)


def filter_split(w_array, stride, out_num, in_num, row, col):
    """ Kernel partition for deconvolution
        list operation

    Parameters
    ----------
    w_array : numpy array
        Input array, of shape (filter_num, map_num, row_num, col_num)

    stride : integer

    out_num : integer

    in_num : integer

    row : integer

    col : integer

    Returns
    -------
    trans_w_array : numpy array
        Transformed weight array of shape (split, filter_num, map_num, row_num, col_num)
    """
    # kernel output channels
    original_filter_num = out_num
    # kernel input channels
    original_map_num = in_num
    # kernel input size
    original_row_num = row
    original_col_num = col

    # partition kernel size
    spilt_row_num = math.ceil(original_row_num / stride)
    spilt_col_num = math.ceil(original_col_num / stride)
    spilt_filter_num = stride ** 2  # partition number

    trans_w_array = np.zeros((spilt_filter_num, original_filter_num, original_map_num, spilt_row_num, spilt_col_num),
                             dtype=float)

    w_array = pad_4d_data(w_array, 0, 0, 0, 0, spilt_col_num * stride - col, 0, spilt_row_num * stride - row, 0)
    # transform
    for cur_spilt_num in range(spilt_filter_num):
        for cur_filter_num in range(original_filter_num):
            for cur_map_num in range(original_map_num):
                for cur_row_num in range(spilt_row_num):
                    for cur_col_num in range(spilt_col_num):
                        trans_w_array[cur_spilt_num,
                                      cur_filter_num, cur_map_num, spilt_row_num - cur_row_num - 1,
                                      spilt_col_num - cur_col_num - 1] = w_array[
                            cur_filter_num, cur_map_num, stride * cur_row_num + cur_spilt_num // stride,
                            stride * cur_col_num + cur_spilt_num % stride]

    return trans_w_array


def image_process(in_array, file_name=""):
    size_y = in_array.shape[0]
    size_x = in_array.shape[1]
    im = Image.new("RGB", (size_y, size_x))

    for i in range(size_x):
        for j in range(size_y):
            im.putpixel((j, i),
                        (int((in_array[i][j][0] + 1) * (255 / 2)), int((in_array[i][j][1] + 1) * (255 / 2)),
                         int((in_array[i][j][2] + 1) * (255 / 2))))

    plt.imsave("./images/" + file_name + ".png", im)


def insert_zeros(in_array, stride, lp=0, rp=0, tp=0, bp=0):
    """ insert_zeros

        Parameters
        ----------
        in_array : numpy array
            Input array, of shape (size_x, size_y, in_channels)

        stride : integer

        """

    size_y = in_array.shape[0]
    size_x = in_array.shape[1]
    in_channels = in_array.shape[2]
    nums_inserted_zero = stride - 1
    assert 0 <= nums_inserted_zero, "nums_inserted_zero can not be negative"

    new_pad_array = np.zeros((size_y + (size_y - 1) * nums_inserted_zero, size_x + (size_x - 1) * nums_inserted_zero, in_channels), dtype=float)
    for cur_y in range(size_y):
        for cur_x in range(size_x):
            new_pad_array[cur_y * stride, cur_x * stride, :] = in_array[cur_y, cur_x, :]
    new_pad_array = pad_3d_data(new_pad_array, lp, rp, tp, bp, 0, 0)

    return new_pad_array


def image_comparison(path="./images/"):
    tf_image = np.array(Image.open(path + "image_tf_deconv.png"))
    split_image = np.array(Image.open(path + "image_split_deconv.png"))
    error = np.mean(tf_image - split_image)

    tf_mean, tf_var = np.mean(tf_image), np.var(tf_image)
    split_mean, split_var = np.mean(split_image), np.var(split_image)
    cov = np.mean((tf_image - tf_mean) * (split_image - split_mean))
    c1 = (0.01*255)**2
    c2 = (0.03*255)**2
    ssim_value = ((2 * tf_mean * split_mean + c1) * (2 * cov + c2)) / \
                 ((tf_mean ** 2 + split_mean ** 2 + c1) * (tf_var + split_var + c2))

    print("The error is: ", error)
    print("The SSIM Value is: ", ssim_value)

