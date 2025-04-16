# Vector Quantized Variational Autoencoder (VQ-VAE)

This repository provides an implementation of a **Vector Quantized Variational Autoencoder (VQ-VAE)**, a powerful model for image compression, reconstruction, and generation. The project is designed to be modular and configurable, making it easy to adapt to various datasets and tasks.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Dataset Preparation](#dataset-preparation)
9. [Requirements](#requirements)
10. [Acknowledgments](#acknowledgments)
11. [License](#license)

---

## Introduction

The **Vector Quantized Variational Autoencoder (VQ-VAE)** is a generative model that combines the power of autoencoders with vector quantization. It is particularly effective for tasks like:

- Image compression
- High-quality image reconstruction
- Generative modeling for image synthesis

This implementation is built using **PyTorch** and **PyTorch Lightning**, ensuring scalability and ease of experimentation.

---

## Features

- **Configurable Architecture**: Easily modify the model's architecture and hyperparameters via a YAML configuration file.
- **Vector Quantization**: Implements a codebook for quantizing latent representations, enabling discrete latent spaces.
- **Perceptual Loss**: Uses LPIPS (Learned Perceptual Image Patch Similarity) for high-quality reconstructions.
- **Adversarial Training**: Includes an optional PatchGAN discriminator for adversarial loss.
- **Dataset Flexibility**: Supports custom datasets with configurable image size and file extensions.
- **Training Utilities**: Includes features like learning rate scheduling, gradient clipping, and checkpointing.
- **Exponential Moving Average (EMA)**: Stabilizes training by maintaining a moving average of model weights.

---

## Repository Structure
```
├── autoencoder.py  # Main VQ-VAE model implementation 
├── config.yaml # Configuration file for model and training 
├── dataset.py # Dataset loader for image data 
├── ema.py # Exponential Moving Average (EMA) implementation 
├── quantize.py # Vector quantization module 
├── train.py # Training script 
├── vqlpips.py # VQ-LPIPS loss with discriminator 
├── vae/ 
    │ 
    ├── autoencoder.py # Autoencoder with KL divergence 
    │ 
    ├── discriminator.py # PatchGAN discriminator 
    │ 
    ├── distribution.py # Gaussian distribution utilities 
    │ 
    ├── loss.py # Loss functions (LPIPS, GAN loss, etc.) 
    │ 
    ├── lpips.py # LPIPS perceptual loss implementation 
    │ 
    ├── unet.py # U-Net architecture for encoder/decoder 
    │ 
    └── utils/ 
    │ 
    └── utils.py # Utility functions (e.g., config loading) 

└── Readme.md # Project documentation
```

---

## Installation

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/vq-vae.git
   cd vq-vae

2. Install dependencies:
    ```python
        pip install -r requirements.txt
    ```

3. Verify installation:
    ```python
        python -c "import torch; print(torch.__version__)"
    ```


## Configuration
The model and training parameters are defined in the config.yaml file. Below is an overview of the key sections:

Model Configuration
```
    model:
    base_learning_rate: 4.5e-06
    target: VQ.autoencoder.VQModel
    params:
        embed_dim: 8
        n_embed: 16383
        ddconfig:
        double_z: False
        z_channels: 8
        resolution: 256
        in_channels: 3
        out_ch: 3
        ch: 128
        ch_mult:
            - 1
            - 1
            - 2
            - 2
            - 4
        num_res_blocks: 2
        attn_resolution:
            - 16
        dropout: 0.0
```

Dataset Configuration
```
    data:
    dataset:
        image_folder: "E:\\YouTube\\stable-diffusion\\dataset\\cat_dog_images"
        image_size: [256, 256]
        extension: [".jpg", ".jpeg", ".png", ".bmp"]
    dataloader:
        batch_size: 2
        shuffle: True
        num_workers: 4
    transform:
        normalize_mean: [0.5]
        normalize_std: [0.5]
        random_flip: True
```

## Training
To train the model, use the train.py script:

```
    python train.py
```

Training Workflow
    1. Loads the configuration from config.yaml.
    2. Initializes the dataset and dataloaders.
    3. Trains the VQ-VAE model using PyTorch Lightning.
    4. Saves checkpoints and logs metrics for visualization.


## Evaluation
To evaluate the model, you can use the log_images method in the VQModel class. This method visualizes reconstructed images and compares them with the input images.

```python
    from autoencoder import VQModel

    # Load the trained model
    model = VQModel.load_from_checkpoint("path/to/checkpoint.ckpt")

    # Evaluate on a sample batch
    reconstructed_images = model.log_images(batch)
```

## Dataset Preparation
Ensure your dataset is organized as follows:
```
    dataset/
    ├── train/
    │   ├── cat1.jpg
    │   ├── dog2.jpg
    │   └── ...
    ├── val/
    │   ├── dog1.jpg
    │   ├── cat2.jpg
    │   └── ...
```

Update the `image_folder` path in `config.yaml` to point to your dataset directory.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- PyTorch Lightning
- torchvision
- tqdm
- numpy
- Pillow
- pyyaml

Install all dependencies using:

```python
    pip install -r requirements.txt
```

## Acknowledgments
This implementation is inspired by the original VQ-VAE paper and incorporates ideas from various open-source projects. Special thanks to the PyTorch and PyTorch Lightning communities for their excellent tools and resources.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

