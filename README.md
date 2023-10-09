# pix2pix-torchLight

This repository hosts an implementation of the Pix2Pix Generative Adversarial Network (GAN) aimed at transforming semantic segmentation labels into realistic-looking images. 

Reference: [Original Pix2Pix Repository](https://github.com/phillipi/pix2pix)

## Model Architecture

### Generator:

The generator consists of an encoder and a decoder with the following layers:

#### Encoder:
- C64
- C128
- C256
- C512
- C512
- C512
- C512
- C512

#### Decoder:
- CD512
- CD512
- CD512
- C512
- C256
- C128
- C64

### Discriminator:

The discriminator architecture is as follows:
- C64
- C128
- C256
- C512

## Hyperparameters

All hyperparameters used in this project can be found in the config file.

## Results

![Description of Image](URL_TO_YOUR_IMAGE)
