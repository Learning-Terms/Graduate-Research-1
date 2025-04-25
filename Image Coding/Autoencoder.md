# Autoencoder

## 🧠 What is an Autoencoder?
- An Autoencoder is a type of **neural network** used to **compress and reconstruct data** — usually **unsupervised (no labels required)**.

![Autoencoder](https://i-blog.csdnimg.cn/blog_migrate/708a6a4d75c2e7e6178ba99b6d3239fc.png)


[CSDN Blog1](https://blog.csdn.net/a13545564067/article/details/139982318?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522e44955dde814ff36f54716eb661dc0b3%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=e44955dde814ff36f54716eb661dc0b3&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-139982318-null-null.142^v102^pc_search_result_base4&utm_term=%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8&spm=1018.2226.3001.4187)


[一文读懂自编码器(AutoEncoder)](https://blog.csdn.net/hellozhxy/article/details/131184241?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522e44955dde814ff36f54716eb661dc0b3%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=e44955dde814ff36f54716eb661dc0b3&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-2-131184241-null-null.142^v102^pc_search_result_base4&utm_term=%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8&spm=1018.2226.3001.4187)

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

## Types of Autoencoder
| Type                     | Main Idea                               | Best For                              |
|--------------------------|------------------------------------------|----------------------------------------|
| Vanilla Autoencoder      | Reconstruct input                        | Basic feature learning                 |
| Denoising Autoencoder    | Reconstruct clean input from noisy       | Noise robustness, denoising            |
| Sparse Autoencoder       | Enforce sparsity in latent layer         | Interpretable features                 |
| Variational Autoencoder  | Learn a distribution in latent space     | Generation, sampling                   |
| Contractive Autoencoder  | Penalize sensitivity to input noise      | Local feature robustness               |
| Convolutional AE         | Use CNNs for encoding/decoding           | Image tasks                            |
| Recurrent Autoencoder    | Use RNNs/LSTMs for sequences             | Time-series, speech, NLP               |
| Seq2Seq Autoencoder      | Encode and decode variable-length input  | NLP, translation, sequence modeling    |
| Adversarial Autoencoder  | Autoencoder + GAN training               | Structured latent space                |
| Attention AE             | Add attention for context sensitivity    | Language/image understanding           |
| Beta/Info/Factor VAE     | Disentangled latent variables            | Generative modeling, interpretability  |



## 📦 Usage of Autoencoders
### 1. 🗜️ Image Compression
- Learn how to **compress images** into smaller binary codes or latent vectors.
- Unlike JPEG, it's learned from data and can preserve semantic features.

### 2. 📈 Dimensionality Reduction(Feature Extraction)
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

## 自编码器的实现示例（使用TensorFlow和Keras）
- 下面是一个使用TensorFlow实现自编码器的简单示例。这个示例展示了如何构建一个基本的自编码器，用于图像数据的压缩和重构。我们将使用经典的MNIST手写数字数据集。

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 加载MNIST数据集
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 将数据展开为一维向量
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# 定义编码器
input_dim = x_train.shape[1]
encoding_dim = 32  # 压缩后的维度

input_img = layers.Input(shape=(input_dim,))
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)

# 定义解码器
decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

# 构建自编码器模型
autoencoder = models.Model(input_img, decoded)

# 构建单独的编码器模型
encoder = models.Model(input_img, encoded)

# 构建单独的解码器模型
encoded_input = layers.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = models.Model(encoded_input, decoder_layer(encoded_input))

# 编译自编码器模型
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# 训练自编码器
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# 使用编码器和解码器进行编码和解码
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# 可视化结果
n = 10  # 显示10个数字
plt.figure(figsize=(20, 4))
for i in range(n):
    # 原始图像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 重构图像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
```

说明：
- 数据预处理: 加载MNIST数据集，并将其像素值归一化到[0,1]区间。
- 模型构建:
  - 编码器: 输入层连接到一个隐藏层（编码层），将数据压缩到32维。
  - 解码器: 将编码后的数据重构回原始维度。
  - 自编码器: 编码器和解码器组合在一起形成完整的自编码器模型。
- 训练模型: 使用binary_crossentropy损失函数和adam优化器进行训练。
- 结果可视化: 显示原始图像和重构图像，以比较它们的相似性。

输出：
![Output](https://i-blog.csdnimg.cn/blog_migrate/c6b1dffe4508db7d80017967d3574498.png)



