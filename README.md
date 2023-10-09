# pix2pix-torchLight

This repository hosts an implementation of the Pix2Pix Generative Adversarial Network (GAN) aimed at transforming semantic segmentation labels into realistic-looking images. 

Reference: [Original Pix2Pix Repository](https://github.com/phillipi/pix2pix)

## Model Architecture

### Generator:

The generator consists of an encoder and a decoder with the following layers:

#### Encoder:
- C64-C128-C256-C512-C512-C512-C512-C512

#### Decoder:
- CD512-CD512-CD512-C512-C256-C128-C64

### Discriminator:

The discriminator architecture is as follows:
- C64-C128-C256-C512

## Hyperparameters

All hyperparameters used in this project can be found in the config file.

## Planned Features and Enhancements

Here are the features and enhancements I'm planning to add to this project in the future:

- [ ] Add weight initialization.
- [ ] Integrate with Weights and Biases (wandb).
- [ ] Add consistency loss function.
- [ ] Implement gradient penalties like WGAN-GP.
- [ ] Implement Frechet Inception Distance (FID).
- [ ] Implement ResnetBlock and ResnetGenerator for StyleGAN adaptation.
- [ ] Upgrade Discriminator:
  - [ ] Use different learning rates.
  - [ ] Add noise to the real image.
  - [ ] Implement early stopping.
  - [ ] Implement PatchGAN with 70x70 patches.

Note: If a checkbox is checked, it indicates that the feature or enhancement has been implemented.


## Results
Left: Generated, Right: Semantic Mask

![Left: Generated, Right: Semantic Mask](https://github.com/xXCoffeeColaXc/pix2pix-torch/blob/master/cherry_picked_results/y_gen_86.png)
![Left: Generated, Right: Semantic Mask](https://github.com/xXCoffeeColaXc/pix2pix-torch/blob/master/cherry_picked_results/y_gen_37.png)
![Left: Generated, Right: Semantic Mask](https://github.com/xXCoffeeColaXc/pix2pix-torch/blob/master/cherry_picked_results/y_gen_39.png)
![Left: Generated, Right: Semantic Mask](https://github.com/xXCoffeeColaXc/pix2pix-torch/blob/master/cherry_picked_results/y_gen_149.png)
![Left: Generated, Right: Semantic Mask](https://github.com/xXCoffeeColaXc/pix2pix-torch/blob/master/cherry_picked_results/y_gen_142.png)

