# Split-Deconvolution
This repo mainly includes two parts. In the first part, we present the python script that 
converts the deconvolution to standard convolution. Then we compare the results calculated using 
the split deconvolution and that calculated using Tensorflow transpose function. The comparison 
verifies the correctness of split deconvolution. 

In the second part, we provide the detailed steps to deploy the split deconvolution on Google 
Edge TPU and NCS2. Meanwhile, we compare the runtime of the baseline deconvolution and the split 
deconvolution, which demonstrates the significant performance speedup of split deconvolution 
over the baseline implementation. Since the native deconvolution (transpose convolution) is not 
supported on Google Edge TPU, we implement the deconvolution using the well-konwn zero padding 
on input activations which is also included in the repo. 

## Prerequisites
* Pillow
* matplotlib
* tensorflow 1.12.0


## Split deconvolution verification
In order to use execute the neural networks of this repo on Tensorflow or Split Deconvolution, you need to provide
* Network configuration `.csv`
* Model Parameters `.npy`
They are included in the repo (`./networks_configuration/` and `./raw_data/`).

1.Calculate deconvolution using transpose function in Tensorflow.
```bash
python Setup.py --model DCGAN --mode tf_deconv
```

2.Convert deconvolution to standard convolution using the proposed split deconvolution
```bash
python Setup.py --model DCGAN --mode split_deconv
```
3.Results comparison
```bash
python Setup.py --model DCGAN --mode verify
```

## Performance comparison 
The example models are included in `./example_models/`. To compare the deconvolution implementations on Google Edge TPU and NCS2, we have one deconvolution layer implemented using both baseline deconvolution and the split deconvolution. The data for the two deconvolution implementations are stored as two `.pb` files which is required by Google Edge TPU  and two `.onnx` files for NCS2. They can be downloaded from the dropbox link https://www.dropbox.com/sh/qenwhtupqkfsezv/AACRLlnFvzCe2VvkXCC3DChJa?dl=0.   

The reason that we do not implement the whole neural network is that the converted deconvolution using both zero padding and split deconvolution need output reorganization for the computing in the next layer. The output reorganization itself is trivial becasue it is essentially to store the output data in the on-chip buffers in DRAM on some scattered but sequential locations of DRAM which is slightly differently to a conventional sequential write back. As DMA module is usually required for any CNN processors because they want to have the output written back to external memory. However, it is not open to users, so we have no choice but to do it on the host. Also the overhead of the data reorganization on host can be measured. The overhead is negligible according to our experiments. The whole dataflow is also verified on our in-house AI chip, through it is not in the market yet. We are working toward to an open sourced FPGA version with netlist. It will appear soon. We will have more experiments announced later.

1. Deconvolution execution on Google Edge TPU  
The inference time of baseline deconvolution (`tf_model.pb`) and split deconvolution (`sd_mdoel.pb`) on TPU is *104.88 ms* and the *49.85 ms* respectively. Note that the output reorganization time is considered in the measurement.  
Detailed instructions to deploy the models on Google Edge TPU can be found in the official documents. https://coral.withgoogle.com/docs/accelerator/get-started/ 

2. Deconvolution execution on NCS2  
The inference time of baseline deconvolution (`transpose_conv.onnx`) and split deconvolution (`split_deconv.onnx`) on TPU is *30.713ms* and the *29.384ms* respectively. Note that the output reorganization time is considered in the measurement.
Detailed instructions can be found in https://software.intel.com/en-us/articles/get-started-with-neural-compute-stick

&nbsp;

To help the users to experiment with their own data, we also provide some auxiliary functions. They are included in `utils/utils.py`.  
`generate_input()` is used to generate input for DCGAN.  
`filter_split()` is used to split and convert the original deconvolution filter.  
`insert_zeros()` is used to insert zeros in the input feature maps for baseline zero-padding-based deconvolution.  
