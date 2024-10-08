# PRODIGY_GA_4
# Pix2Pix: Image-to-Image Translation with Conditional Generative Adversarial Networks

## Overview

This project implements a conditional Generative Adversarial Network (cGAN) using TensorFlow and Keras to generate images from the CIFAR-10 dataset. The GAN consists of two models: a Generator and a Discriminator, which are trained simultaneously to improve image quality.

## Prerequisites

- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- NumPy
- tqdm

## Installation

You can install the required packages using pip:

```bash
pip install tensorflow keras matplotlib numpy tqdm
```
## Dataset

The model uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes:  
- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

The dataset is commonly used for image classification and provides a standard benchmark for evaluating machine learning models.

## Usage

1. **Load the Dataset**: The CIFAR-10 dataset is loaded and normalized within the code. The images are scaled to a range of [-1, 1] to improve model training.

2. **Define Models**: The code defines a Generator and a Discriminator. The Generator creates new images, while the Discriminator evaluates whether the images are real or generated.

3. **Training**: To train the GAN, run the `train()` function. You can specify the number of epochs to control the training duration.

4. **Generate Samples**: After training, you can generate new images by calling the `show_samples()` function, which visualizes generated images for each class based on random noise and class labels.
## Training

The training process can be configured by modifying the following parameters in the code:

- **`batch_size`**: Defines the size of the training batches. A larger batch size can speed up training but requires more memory.
  
- **`epoch_count`**: Specifies the number of training epochs. Increasing this value allows the model to learn more, but training time will also increase.
  
- **`noise_dim`**: Represents the dimension of the random noise input to the generator. This controls the diversity of generated images.
  
- **`n_class`**: Indicates the number of classes in the dataset. This is set based on the specific dataset being used (e.g., 10 for CIFAR-10).


### Example

To train the model and generate images, include the following in your code:

```python
train(dataset, epochs=epoch_count)
```

