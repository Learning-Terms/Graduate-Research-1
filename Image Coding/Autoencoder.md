# Autoencoder

## 🧠 What is an Autoencoder?
- An Autoencoder is a type of **neural network** used to **compress and reconstruct data** — usually **unsupervised (no labels required)**.

- It has two main parts: **Encoder&Decoder**

```css
  [ Input Image ]
     ↓
   Encoder      🔒 (compresses the image into a smaller vector)
     ↓
 Latent Code    🧠 (compact representation — like "compressed meaning")
     ↓
   Decoder      🔓 (reconstructs the image from the latent code)
     ↓
[ Reconstructed Image ]
```

## 🎯 Purpose
- The network **learns to keep only the most important information** to **rebuild the original data as closely as possible**
— which is exactly **what compression is about!**

## 🔧 Key Components:
- **Encoder**	Maps input to a compressed latent space (e.g., convolutional layers)
- **Latent Space**	Compressed representation of the data
- **Decoder**	Reconstructs the input from the latent vector

## 📦 Usage of Autoencoders
### 1. 🗜️ Image Compression
- Learn how to **compress images** into smaller binary codes or latent vectors.
- Unlike JPEG, it's learned from data and can preserve semantic features.

### 2. 📈 Dimensionality Reduction
- Like PCA but **non-linear and learned via deep learning**.
- Useful for visualizing or simplifying high-dimensional data.

### 3. 🎭 Denoising
- Train on clean images, and input noisy ones — the model learns to "clean" them.
- Used in medical imaging, photos, and signal processing.

### 4. 🧬 Anomaly Detection
- Train on normal data → model learns what "normal" looks like.
- If reconstruction error is high, it might be an anomaly.

### 5. 🎨 Image Generation
- Part of Variational Autoencoders (VAEs) or GANs for generating new images.

## Example (Image Compression with Autoencoder)
Imagine compressing a 128×128 image into a 16-dimensional latent vector:
- Encoder shrinks it down.
- Decoder rebuilds it from the 16 values.
- If trained well, **the output image looks very similar — but the size is tiny**!

## Repositories from Github
### 🖼️ Image Compression with Autoencoders
#### Learned Image Compression Using Autoencoder Architecture
- Description: Demonstrates a basic architecture for learned image compression, discussing main building blocks, hyperparameters, and comparisons to traditional codecs.
- Repository: [Github](https://github.com/MahmoudAshraf97/AutoencoderCompression?utm_source=chatgpt.com)

#### Image Compression using CNN-based Autoencoders
- Description: Showcases how deep learning can compress images to very low bitrates while retaining high quality, using CNN-based autoencoders.
- Repository: [Github](https://github.com/abskj/lossy-image-compression?utm_source=chatgpt.com)

### 🚨 Anomaly Detection with Autoencoders
#### Anomaly Detection with Autoencoder
- Description: Utilizes autoencoders trained on normal data to identify anomalies that deviate from learned patterns.
- Repository: [Github](https://github.com/AarnoStormborn/anomaly-detection-with-autoencoder?utm_source=chatgpt.com)

#### Anomaly Detection using Autoencoders
- Description: Explores how autoencoders can detect anomalies by learning compressed representations of normal data and identifying deviations.
- Repository: [Github](https://github.com/hellomlorg/Anomaly-Detection-using-Autoencoders?utm_source=chatgpt.com)

### 🧼 Image Denoising with Autoencoders
#### UNet-based Denoising Autoencoder in PyTorch
- Description: Cleans printed text using a denoising autoencoder based on the UNet architecture, implemented in PyTorch.
- Repository: [Github](https://github.com/n0obcoder/UNet-based-Denoising-Autoencoder-In-PyTorch?utm_source=chatgpt.com)

#### Denoising Autoencoder - Udacity Deep Learning v2 PyTorch
- Description: Demonstrates denoising autoencoders using the MNIST dataset, adding noise to data and training the model to reconstruct clean images.
- Repository: [Github](https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/denoising-autoencoder/Denoising_Autoencoder_Solution.ipynb?utm_source=chatgpt.com)





