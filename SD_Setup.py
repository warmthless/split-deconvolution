import SD_Inference
import numpy as np
from utils.utils import image_process
import argparse


class SplitDeconvolution:
    def __init__(self, network_name, config):
        self.current_networks = network_name
        self.network_parameters = config

    def run_sim(self):

        network_parameters_file = open(self.network_parameters, 'r')
        first_row_name = True
        latest_result = np.load("./raw_data/" + self.current_networks + "/input/Input_Data.npy")
        for row in network_parameters_file:
            if first_row_name:
                first_row_name = False
                continue

            layer_parameters = row.strip().split(',')
            if len(layer_parameters) < 10:

                continue

            name = layer_parameters[0]
            print("Running for " + name + "\n")

            weight_file_path = "./raw_data/" + self.current_networks + "/weight/" + name + "_weight.npy"
            bias_file_path = "./raw_data/" + self.current_networks + "/bias/" + name + "_bias.npy"

            # network parameters
            stride = int(layer_parameters[1])
            output_x_length = int(layer_parameters[2])
            output_y_length = int(layer_parameters[3])
            input_pad = [int(layer_parameters[4]), int(layer_parameters[5]),
                         int(layer_parameters[6]), int(layer_parameters[7])]
            op_mode = layer_parameters[8]
            act_mode = layer_parameters[9]

            running = SD_Inference.Computation(stride, output_x_length,
                                               output_y_length, input_pad, op_mode, act_mode,
                                               latest_result, weight_file_path, bias_file_path)
            latest_result = running.operation_executive()
        image_process(latest_result)

        print("Split Deconvolution Inference Complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', help='Network Name', required=True)
    parser.add_argument(
        '--model', help='Network Config', required=True)

    args = parser.parse_args()
    SD = SplitDeconvolution(args.name, args.model)
    SD.run_sim()


if __name__ == '__main__':
    main()

