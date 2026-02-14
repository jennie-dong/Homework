# Deep Learning Projects

A collection of three deep learning projects implemented in PyTorch, covering transfer learning, neural style transfer, and object detection.

## Project 1: Transfer Learning with Vision Transformer

**Goal:** Classify CIFAR-100 images (100 classes) using a pre-trained Vision Transformer (ViT-B/16).

- **Model:** ViT-B/16 pre-trained on ImageNet-1K, with the classification head replaced (768 → 100)
- **Dataset:** CIFAR-100 — 50,000 training / 10,000 test images, resized to 224×224
- **Augmentation:** Random cropping (padding=8), random horizontal flip
- **Optimizer:** Adam (lr=1e-4), cross-entropy loss, batch size 64
- **Results:** Achieved **83.50% validation accuracy** after 2 epochs of fine-tuning
- **Observations:** Strong performance on visually distinct classes (table, tiger, turtle); confusion on morphologically similar classes (possum vs. rabbit)

## Project 2: Neural Style Transfer

**Goal:** Transfer the artistic style of a landscape painting onto a photograph of the Sydney Opera House using VGG19.

- **Model:** Pre-trained VGG19 (frozen); feature extraction at layers 3, 8, 15, 22
- **Loss:** Content loss (MSE at layer 22) + Style loss (Gram matrix MSE across all selected layers); style weight = 1e7, content weight = 1
- **Optimization:** Adam (lr=0.01), directly optimizing pixel values over 1,000 iterations
- **Results:** Loss decreased from 601.33 → 8.35; output preserves the Opera House geometry while applying impressionistic color palette and brushstroke patterns

## Project 3: YOLOv5 Object Detection

**Goal:** Train YOLOv5 for real-time object detection on a Peking University campus video.

- **Model:** YOLOv5n (nano, 1.77M parameters, 4.2 GFLOPs), fine-tuned from pre-trained weights
- **Dataset:** 3,971 frames (640×480) from campus video; 4 classes — person, bicycle, car, motorcycle; split into 1,800 training / 2,172 validation images
- **Training:** 50 epochs, batch size 8, SGD (lr=0.01), image size 640×640, on NVIDIA RTX PRO 6000
- **Results:**
  - box_loss: 0.095 → 0.03; cls_loss: 0.037 → <0.01
  - Inference speed: ~1.9ms per image — suitable for real-time applications
  - Successfully detected and annotated objects across all 3,971 frames

## Tech Stack

- **Frameworks:** PyTorch, TorchVision
- **Models:** ViT-B/16, VGG19, YOLOv5n
- **Libraries:** OpenCV, NumPy, Matplotlib, Pandas
- **Hardware:** NVIDIA RTX PRO 6000, CUDA
