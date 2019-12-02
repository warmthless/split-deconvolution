import numpy as np
import utils.utils as util
import math


class Computation:
    def __init__(self, stride, output_x_length, output_y_length, input_pad, op_mode,
                 act_mode, input_data, weight_file_path, bias_file_path):

        # networks parameters
        self.stride = stride
        self.output_x_length = output_x_length
        self.output_y_length = output_y_length
        self.input_pad = input_pad
        self.op_mode = op_mode
        self.act_mode = act_mode

        self.input_data = input_data
        self.weight = np.load(weight_file_path)
        self.bias = np.load(bias_file_path)

    def gan_fc(self):
        output = np.dot(self.input_data, self.weight) + self.bias
        output = self.activation_function(output)
        reshape_channels = int(self.weight.shape[1]/(self.output_x_length*self.output_y_length))

        return output.reshape((self.output_x_length, self.output_y_length, reshape_channels))

    def split_deconvolution(self):
        self.input_data = util.pad_3d_data(self.input_data, self.input_pad[0], self.input_pad[1],
                                           self.input_pad[2], self.input_pad[3], 0,  0)
        output_channels, input_channels, filter_y, filter_x = self.weight.shape[0], self.weight.shape[1], \
                                                              self.weight.shape[2], self.weight.shape[3]

        # split filter
        self.weight = util.filter_split(self.weight, self.stride, output_channels, input_channels, filter_y, filter_x)

        split_nums = self.weight.shape[0]
        filter_y, filter_x = self.weight.shape[3], self.weight.shape[4]
        deconv_output = np.zeros((self.output_y_length, self.output_x_length, output_channels), dtype=float)
        split_output_y, split_output_x = int(self.output_y_length / self.stride), int(self.output_x_length / self.stride)

        for cur_split_group in range(split_nums):
            # Convolution Nest Loop
            for cur_oc in range(output_channels):
                for cur_oy in range(split_output_y):
                    for cur_ox in range(split_output_x):

                        deconv_y = cur_oy * self.stride + math.floor(cur_split_group / self.stride)
                        deconv_x = cur_ox * self.stride + int(math.fmod(cur_split_group, self.stride))

                        for cur_ky in range(filter_y):
                            for cur_kx in range(filter_x):
                                cur_input = self.input_data[cur_oy + cur_ky, cur_ox + cur_kx, :]
                                cur_weight = self.weight[cur_split_group, cur_oc, :, cur_ky, cur_kx]
                                deconv_output[deconv_y, deconv_x, cur_oc] += np.dot(cur_input, cur_weight)

                        deconv_output[deconv_y, deconv_x, cur_oc] += self.bias[0, cur_oc]

        deconv_output = self.activation_function(deconv_output)

        return deconv_output

    def operation_executive(self):
        if self.op_mode.strip() == "gan_fc":
            return self.gan_fc()
        elif self.op_mode.strip() == "gan_deconv":
            return self.split_deconvolution()

    def activation_function(self, array):
        if self.act_mode.strip() == "relu":
            array[array < 0] = 0
        elif self.act_mode.strip() == "tanh":
            for in_channel in range(array.shape[2]):
                for y in range(array.shape[1]):
                    for x in range(array.shape[0]):
                        array[x, y, in_channel] = math.tanh(array[x, y, in_channel])
        return array




