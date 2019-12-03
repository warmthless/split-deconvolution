# Split-Deconvolution
This repo mainly includes two parts. In the first part, we present the python script that 
converts the deconvolution to standard convolution. Then we compare the results calculated using 
the split deconvolution and that calculated using Tensorflow transpose function. The comparison 
verfies the correctness of split deconvolution. 

In the second part, we provide the detailed steps to deploy the split deconvolution on Google 
Edge TPU. Meanwhile, we compare the runtime of the baseline deconvolution and the split 
deconvolution, which demonstrates the significant performance speedup of split deconvolution 
over the baseline implementation. Since the native deconvolution (transpose convolution) is not 
supported on Google Edge TPU, we implement the deconvolution using the well-konwn zero padding 
on input activations which is also included in the repo. 

## Prerequisites
* Pillow
* matplotlib
* tensorflow 1.12.0


## Split deconvolution verification
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

## Deconvolution execution on Google Edge TPU
The code can be found in ./runtime-comparison
Platform: 


In order to use your own data on Google Edge TPU, you need to provide
* Network configuration `.csv`
* Model Parameters `.npy`

We implement some auxiliary functions to help with the deployment. 
They are included in `utils/utils.py`.  
`generate_input()` is used to generate input for DCGAN.  
`filter_split()` is used to split and transform the original deconvolution filter.  
`insert_zeros()` is used to insert zeros in the input feature maps for native zero-padding deconvolution.  

## TPU Models
There are two `.pb` files of one deconvolutional layer for Edge TPU since the reorganization of the feature maps need to be done on the host.
The inference time of `tf_model.pb` and `sd_mdoel.pb` on TPU is *8.346 ms* and the *2.398 ms* including the time of reorganization on the host using cpp script.
For the deployment of models on Edge TPU, you can follow the official documents. https://coral.withgoogle.com/docs/accelerator/get-started/ 

## Dropbox
This is the Dropbox link https://www.dropbox.com/sh/qenwhtupqkfsezv/AACRLlnFvzCe2VvkXCC3DChJa?dl=0
