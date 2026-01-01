📌 Overview

This project implements a Brain Tumor Detection system using deep learning techniques for medical image segmentation and classification.

The work is centered around U-Net and Attention U-Net architectures, with an additional classification head attached to the encoder. Multiple training strategies and design choices are explored and analyzed.

The entire implementation is provided in a single, well-documented Jupyter Notebook, making it easy to run, understand, and present.

✨ Features

✅ Brain tumor segmentation using U-Net

✅ Attention U-Net for improved segmentation performance

✅ Encoder-based classification head

✅ Joint vs separate training analysis

✅ Multiple classifier architecture comparisons

✅ Hyperparameter tuning experiments

✅ End-to-end inference & visualization pipeline

✅ Clean, presentation-ready notebook

📂 Dataset

BRISC 2025 – Brain Tumor Dataset

🔗 Download from Kaggle:
https://www.kaggle.com/datasets/briscdataset/brisc2025

📁 Dataset Setup

Download the dataset from Kaggle.

Extract all files.

Place all dataset folders/files in the same directory as the .ipynb notebook.

⚠️ Important:
No path changes are required. The notebook assumes the dataset is located in the same folder.

🚀 How to Run
1️⃣ Clone the Repository
git clone https://github.com/your-username/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection

2️⃣ Install Dependencies
pip install numpy matplotlib opencv-python scikit-learn torch torchvision jupyter


(Optional: create a virtual environment for best practice.)

3️⃣ Run the Notebook
jupyter notebook


Open the .ipynb file

Run all cells sequentially

No manual configuration needed

✅ The notebook will:

Load and preprocess data

Train segmentation and classification models

Evaluate performance

Visualize predictions

🏗️ Model Architectures
🔹 U-Net (Baseline)

Encoder-decoder architecture with skip connections

Used for pixel-level tumor segmentation

🔹 Attention U-Net

Adds attention gates to skip connections

Helps the model focus on tumor-relevant regions

Trained and compared against baseline U-Net

🔹 Classification Head

Attached to the encoder output

Encoder features are pooled and passed to a classifier

Multiple classifier designs tested

🔬 Experiments & Analysis
✔ Joint vs Separate Training

Segmentation and classification heads trained:

Jointly (multi-task learning)

Separately (encoder feature extraction)

Performance comparison and analysis included

✔ Classifier Architecture Comparison

Fully Connected heads

Deeper MLP classifiers

CNN-based classifier variants

✔ Hyperparameter Tuning

Optimizers tested: Adam, SGD, RMSprop

Wide learning-rate search

Batch size and regularization experiments

📊 All results are compiled and discussed in the notebook.

🖼️ Inference & Visualization

The notebook includes a complete inference pipeline that:

Takes any MRI image as input

Displays:

Original image

Predicted segmentation mask

Predicted tumor class

This makes the project ideal for demonstrations and evaluations.

❌ Not Included

🚫 EfficientDet-based decoder integration (High-difficulty bonus task)

🛠️ Tech Stack

Python

PyTorch

NumPy

OpenCV

Matplotlib

Scikit-learn

Jupyter Notebook

📑 Notes

A single unified preprocessing pipeline is used for all images.

Notebook is well-commented and structured for easy understanding.

No separate presentation slides are required.

🙌 Acknowledgments

Dataset: BRISC 2025 (Kaggle)

Model inspiration:

U-Net: Ronneberger et al.

Attention U-Net: Oktay et al.

⭐ Support

If you find this project helpful, consider giving it a ⭐ star on GitHub!
