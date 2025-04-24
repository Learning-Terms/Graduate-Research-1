# Image Coding for Machines with Object Region Learning

## Abstract
### What is the research topic?

The research topic is **Image Coding for Machines (ICM)** — a subfield of **image compression** that aims to **optimize image encoding** not just for human viewing, but **specifically for machine perception and recognition tasks**, such as **object detection and image segmentation**.

The core proposal of the paper is **a novel image compression model** that:
- Learns object regions automatically (without requiring extra inputs like **an ROI-map**).
- Does **not rely on task-loss**, making it model-agnostic and **applicable to various image recognition systems**.

### What techniques are needed in this research?
#### ✅ Image Compression Techniques
- Understanding of traditional and modern **image compression principles and algorithms**.
- Knowledge of how to design or modify compression models, likely using deep learning.

#### ✅ Deep Learning
- Use of deep neural networks to **learn representations of object regions automatically**.
- Training and evaluation of deep models **without relying on task-specific loss functions**.

#### ✅ Object Detection and Image Segmentation
- Understanding of how machines **interpret images through models like YOLO, Mask R-CNN**, or others.
- Use of these models as **downstream tasks** to validate the effectiveness of the compression method.

#### ✅ ICM-Specific Strategies
- There are two main approaches in ICM: **the ROI-based approach and the taskloss-based approach**.
- The former approach has the problem of requiring an ROI-map as input in addition to the input image.The latter approach has the problems of difficulty in learning the task-loss, and lack of robustness because **the specific image recognition model is used to compute the loss function**. To solve these problems, we propose **an image compression model** that learns object regions. Our model does not require additional information as input, such as an ROI-map, and does not use task-loss.
- Familiarity with ROI-based approaches (require region-of-interest maps).
- Understanding of task-loss-based approaches (require differentiable task models).

#### ✅ Model Evaluation Techniques
- Designing experiments with **multiple image recognition models and datasets**.
- Quantitative comparison with existing methods to validate performance, robustness, and versatility.

#### Index Terms—Image coding for Machines, ICM, image compression, object detection,segmentation

### Questions
#### What is the meaning of Image Compression before image recognation?

- **Image Compression before Image Recognition** means **compressing an image first**, then sending it to an image recognition system (like object detection, classification, or segmentation) — instead of feeding the original, full-size image.
- It’s often **necessary to reduce data size before transmitting or processing images on appliactions like Self-driving cars, Surveillance cameras,Mobile devices and Remote sensing (e.g., satellites or drones)**. But if you're compressing images just for machines, not for human viewing, your goal isn’t visual quality — **it’s machine accuracy**.
- Traditional image compression focuses on what looks good to humans (e.g., JPEG, PNG).
But for machines, we care about **preserving features important for recognition** — even if the image looks strange to people.
- So, Image Compression before Image Recognition means:**Designing compression that keeps the important info for AI models.** and **Possibly sacrificing human-visible quality if it helps machine performance.**

#### Does the image recogantion model itself like YOLO has image compression algorithms?
- **❌ No, image recognition models like YOLO do not perform image compression**.
- YOLO (You Only Look Once) is a real-time object detection model. It takes an input image (already in a usable format, like JPEG or a tensor) and **Finds objects, Classifies them and Outputs bounding boxes and labels**.
- Image compression is done before recognition, by a separate process or model, to: **Reduce image size, Remove "unimportant" information, Save storage or speed up transmission**.
- Image compression may be done by other deep compression models like **Autoencoders, Variational Autoencoders (VAEs)**.
- Image compression is not part of YOLO itself, but can be used before YOLO to feed it better, smaller inputs.
- 🔄 So the flow looks like this:
```css
[ Raw Image ]
     ↓
[ Compression Model for Machines ]
     ↓
[ Compressed Image (optimized for recognition) ]
     ↓
[ YOLO or other detection model ]
     ↓
[ Detection Results ]
```

## Introduction
### Captures from paper
- **Image compression**  is essential, especially on occasions where many images need to be **transmitted and stored** while having **limited bandwidth and storage**.
-  image coding methods such as JPEG, AVC/H.264, HEVC/H.265, and VVC/H.266 have been created. These image compression methods are composed of hand-crafted algorithms, created based on the
knowledge of data encoding experts.
- **Neural network based image compression(NIC)** has also been the subject of much research in recent years.
- Conventional coding methods compress images while preserving image quality. In other words, they are **not efficient compression methods for image recognition**. Hence, it is necessary to devise an image compression method **specifically for object detection models and segmentation models**. The research field on image compression for such purposes is called **Image Coding for Machines (ICM)**.
- There are two main approaches in the study of ICM.
     - The first approach is **ROI-based method** in which **ROI-map** is used to allocate more bits to the object region in the images. As shown in Fig. 1(a), this approach needs an **ROI-map as input** in addition to the image to be encoded. The problem with this approach is that it requires process to prepare the ROI-map before compressing the image.
     -  The second approach is **task-loss-based method**. As shown in Fig. 1(b), in this approach, the NIC model is trained using task-loss to create an image compression model for image recognition. The **task-loss is calculated by the image recognition accuracy of the coded image created using the NIC model**. For example, when training an NIC model for object detection, the detection accuracy of the coded image by NIC model is used as the loss function. Thus, the loss function is defined by the output values of the image recognition model. **However, these values are output from a black box, which makes it difficult for the NIC model to learn them**. In addition, when training an NIC model with task-loss, the **NIC model corresponding to the image recognition model is required**. It is due to the variation in task-loss, dependent on the type of image recognition model.
- To solve this problem, we propose an NIC model which **learns the object region in the images**. This **compression model** is trained using a **Object-MSE-loss**. The Object-MSE-loss is **the difference between the object region in the input image and that of the output decoded image**. By applying this loss to train the NIC model, only the object regions is decoded cleanly leaving the other regions untouched. Thus, **information in the image that is unnecessary for image recognition is eliminated**.

