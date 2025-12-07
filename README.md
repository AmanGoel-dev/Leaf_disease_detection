🌿 Cassava Leaf Disease Detection

Deep Learning Project using CNN, EfficientNet-B3, ViT-B/16, and Swin-L

This project classifies cassava leaf images into 5 categories using multiple deep learning architectures.
The main goal is to help detect plant diseases early and assist farmers in preventing crop loss.

📂 Project Structure

Your current structure:

LEAF_DISEASE/
├── BaseCNN.py
├── EfficientNetB3.py
├── SwinL4th.py
├── vitB16.py
└── README.md  ← (add this file)

File Descriptions
File	Description
BaseCNN.py	Custom CNN baseline model (trained from scratch).
EfficientNetB3.py	Transfer learning model using EfficientNet-B3.
vitB16.py	Vision Transformer (ViT-B/16) implementation.
SwinL4th.py	Swin Transformer Large (patch4-window12-384).
🚀 Models Implemented
1️⃣ Base CNN (Baseline)

15.9M parameters

Image size: 224×224

Validation Accuracy: 76.12%

2️⃣ EfficientNet-B3 (Transfer Learning)

11.5M parameters

Image size: 512×512

Best Validation Accuracy: 84.00%

Best Validation Loss: 0.4684 (best among all)

3️⃣ Vision Transformer – ViT-B/16

86.6M parameters

Global self-attention

Validation Accuracy: 86.48%

4️⃣ Swin Transformer Large (Swin-L)

197M parameters

Window-based multi-head attention

Validation Accuracy: 89.04% (highest)

📊 Dataset

Source: Kaggle – Cassava Leaf Disease Detection

Total images: 21,397

Classes: 5

Challenges:

High class imbalance (CMD = 61%)

Noisy, variable mobile-captured images

Different resolutions

🧪 Preprocessing & Augmentation

All models follow these core steps:

Resize images (224 / 384 / 512 depending on model)

Normalize using ImageNet mean/std

Train-val split (80–20 stratified)

Albumentations augmentations:

Random Rotate

Flip (H/V)

Color Jitter

Gaussian Noise / Blur

Brightness/Contrast

Coarse Dropout

▶️ How to Run
Base CNN
python BaseCNN.py

EfficientNet-B3
python EfficientNetB3.py

Vision Transformer ViT-B/16
python vitB16.py

Swin Transformer Large
python SwinL4th.py


Make sure you have GPU enabled for Swin-L and ViT models — they are heavy!

🛠 Installation

Install required packages:

pip install -r requirements.txt


Recommended libs:

tensorflow
torch
timm
albumentations
opencv-python
numpy
matplotlib
scikit-learn

📈 Model Comparison
Model	Val Accuracy	Val Loss
Base CNN	76.12%	0.7355
EfficientNet-B3	84.00%	0.4684
ViT-B/16	86.48%	0.6907
Swin-L	89.04%	0.3333
📌 Future Improvements

Ensemble (EfficientNet + ViT + Swin)

More real-world cassava leaf images

Deploy using TensorFlow Lite / ONNX

Try ConvNext or DeiT as alternative backbones

👨‍👩‍👦 Contributors

Arshiya Kamal

Prasukh Jain

Bhanavi Goyal

Aman Goel

Paras Dang
