# Split-Deconvolution
This is a demo of the split deconvolution inference process, which also includes the tensorflow inference process for verification.

## Requirements
* Pillow
* matplotlib
* tensorflow 1.12.0


## Inference Demo

```bash
python SD_Setup.py --model DCGAN
```

## Usage

In order to use your own data, you have to provide
* Network configuration `.csv`
* Model Parameters `.npy`

## Utils

There are some function tools in `utils/utils.py`.

`generate_input()` is used to generate input for DCGAN.
`filter_split()` is used to split and transform the original deconvolution filter.
`insert_zeros()` is used to insert zeros in the input feature maps for native zero-padding deconvolution.

## TPU Models
There are two `.pb` files of one deconvolutional layer for Edge TPU since the reorganization of the feature maps need to be done on the host.
The inference time of `tf_model.pb` and `sd_mdoel.pb` on TPU is *8.346 ms* and the *2.398 ms* including the time of reorganization on the host using cpp script.
For the deployment of models on Edge TPU, you can follow the official documents. https://coral.withgoogle.com/docs/accelerator/get-started/ 

## Dropbox
This is the Dropbox link https://www.dropbox.com/sh/qenwhtupqkfsezv/AACRLlnFvzCe2VvkXCC3DChJa?dl=0
