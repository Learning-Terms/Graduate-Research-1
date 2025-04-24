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



