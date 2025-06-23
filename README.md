# Comparative Analysis of Hybrid ML and DL Models for Medical Image Classification

## Description

This project is dedicated to developing advanced medical image classification systems for brain tumors and chest X-rays using standalone machine learning (ML), standalone deep learning (DL), and hybrid ML/DL approaches. Medical image analysis is crucial for early and accurate disease detection, directly impacting patient outcomes and care. By utilizing datasets from Kaggle—consisting of meticulously annotated brain MRI images and chest X-ray images—models are trained to differentiate between normal scans and those displaying tumor or infection anomalies. The project is implemented using Jupyter Notebooks and supports collaborative development and experimentation for ML, DL, and hybrid methodologies in medical image classification.

## Features

- **Standalone ML, DL, and Hybrid Models:** Comprehensive evaluation of Support Vector Machines (SVM), Convolutional Neural Networks (CNN), and hybrid (CNN + SVM/XGBoost) architectures.
- **Dataset Support:**  
  - **Chest X-Ray:** [Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
  - **Brain Tumor:** [Brain Tumor Classification (MRI) Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- **Automated Feature Extraction:** Leverages deep learning for automatic feature extraction and traditional ML for classification.
- **Model Training and Evaluation:** Includes scripts for model training, validation, and performance assessment.
- **Comprehensive Reporting:** Provides a detailed final analysis report with results, visualizations, and insights.
- **Code Organization:** All code, notebooks, and report files are included and clearly linked for easy access.

## Requirements

- **Python 3.6 or higher**
- **pip (Python package installer)**
- **Required Libraries:**
  - TensorFlow and Keras (for deep learning model development)
  - scikit-learn (for traditional ML models and evaluation)
  - NumPy and PIL (for image processing)
  - pandas and matplotlib (for data handling and visualization)
- **GPU (highly preferable for faster training):** You can use Kaggle Notebook or Google Colab for GPU access.
- **Optional:** Jupyter Notebook for interactive development

## Setup Instructions

```bash
#Clone the repository
git clone https://github.com/arundhatigvasishth/Comparitive-Ananlysis-of-Hybrid-ML-and-DL-Models-for-Medical-Image-Classification.git
cd Comparitive-Ananlysis-of-Hybrid-ML-and-DL-Models-for-Medical-Image-Classification

# If you do not have Python 3.6 or higher, create a virtual environment using this in your command prompt

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

# DOWNLOAD ALL THE REQUIRED PACKAGES BY RUNNING:
pip install -r requirements.txt

# YOU CAN RUN THE NOTEBOOKS OR SCRIPTS FOR TRAINING AND ANALYSIS.
# For example, open and run the desired .ipynb file in Jupyter Notebook, or use:
jupyter notebook

# For GPU acceleration, you may also use Kaggle Notebook or Google Colab by uploading the code and data.
