🧠 Brain Tumor Detection using U-Net and Attention U-Net

This repository contains the complete implementation of a Brain Tumor Detection system using deep learning. The project focuses on medical image segmentation and classification using U-Net and Attention U-Net architectures, along with extensive experimentation on training strategies, classifier heads, and hyperparameters.

The implementation is provided as a single, well-documented Jupyter Notebook, designed to be easily reproducible and presentation-ready.

📌 Project Objectives

Perform brain tumor segmentation using U-Net.

Attach a classification head to the encoder for tumor classification.

Compare joint vs separate training of segmentation and classification heads.

Upgrade the model to Attention U-Net and analyze performance improvements.

Experiment with multiple classifier architectures and training hyperparameters.

Provide a unified inference pipeline for visualization and prediction.

📂 Dataset

The dataset used in this project is BRISC 2025 Brain Tumor Dataset, available on Kaggle.

🔗 Download link:
https://www.kaggle.com/datasets/briscdataset/brisc2025

Dataset Setup Instructions

Download the dataset from Kaggle.

Extract the dataset files.

Place all dataset files and folders inside the same directory as the .ipynb notebook.

No additional path configuration is required.

Once placed correctly, the notebook will automatically load the data.

🚀 How to Run the Project

Clone this repository:

git clone https://github.com/your-username/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection


Install the required dependencies:

pip install -r requirements.txt


(If requirements.txt is not provided, required libraries are listed below.)

Open the Jupyter Notebook:

jupyter notebook


Open the .ipynb file.

Run all cells sequentially — no manual intervention required.

✅ The notebook will:

Load and preprocess the data

Train segmentation and classification models

Evaluate performance

Visualize predictions

🏗️ Model Architectures
1. U-Net (Baseline)

Used for pixel-wise tumor segmentation.

Encoder-decoder structure with skip connections.

2. Attention U-Net

Enhances U-Net by integrating attention gates.

Improves focus on tumor-relevant regions.

Trained and evaluated separately for comparison.

3. Classification Head

Attached to the encoder output.

Multiple classifier architectures tested:

Fully Connected Networks

Deeper MLP-based heads

CNN-based classifier variants

🔬 Experiments & Analysis

✔ Joint vs Separate Training

Segmentation and classification heads trained:

Jointly (multi-task learning)

Separately (feature extraction + classifier)

✔ Classifier Architecture Comparison

Performance comparison across different classifier designs.

✔ Hyperparameter Tuning

Optimizers tested: Adam, SGD, RMSprop

Wide learning rate exploration

Batch size and regularization analysis

📊 All results are compiled and discussed inside the notebook.

🖼️ Inference & Visualization

The notebook includes a complete inference pipeline that:

Takes any input MRI image

Displays:

Original image

Predicted segmentation mask

Predicted tumor class

This makes the notebook suitable for live demonstrations and evaluations.

❌ Not Implemented

EfficientDet-based decoder integration (High-difficulty bonus task) is intentionally not included.

🛠️ Technologies Used

Python

PyTorch / TensorFlow (depending on implementation)

NumPy

OpenCV

Matplotlib

Scikit-learn

Jupyter Notebook

📑 Notes

A single preprocessing pipeline is applied to all images.

The notebook is structured and commented for easy understanding.

No separate presentation slides are required due to the clarity of the notebook.

📬 Acknowledgments

Dataset provided by BRISC 2025 via Kaggle.

U-Net and Attention U-Net inspired by original research papers.

⭐ If You Find This Useful

Feel free to ⭐ star the repository and share feedback!
