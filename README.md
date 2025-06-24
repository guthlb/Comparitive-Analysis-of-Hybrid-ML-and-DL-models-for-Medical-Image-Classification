## Description

This project features pneumonia detection by binary classification of Chest X-ray images, and tumor classification of Brain Tumor MRIs using standalone machine learning (ML), standalone deep learning (DL), and hybrid ML+DL approaches. Medical image analysis is crucial for early diagnosis of diseases, directly impacting patient outcomes and care. By utilizing datasets from Kaggle consisting of annotated brain MRI images and chest X-ray images, models are trained to differentiate between normal scans and those displaying tumor or infection anomalies. The project is implemented using Kaggle Notebooks and supports collaborative development and experimentation for ML, DL, and hybrid methodologies in medical image classification.

## Models

#### A. Standalone Machine Learning (ML)
- **Support Vector Machine (SVM)**  
  - RBF and Linear kernels with Principal Component Analysis (PCA) feature extraction
- **Extreme Gradient Boosting (XGBoost) Classifier**  
  - Raw implementation without feature extraction
- **XGBoost with PCA feature extraction**

#### B. Standalone Deep Learning (DL)
- **Convolutional Neural Network (CNN)**  
  - Transfer Learning using VGG16 architecture

#### C. Hybrid ML + DL Approach
- **XGBoost** with Principal Component Analysis (PCA) feature extraction

## Datasets

  - **Chest X-Ray:** [Chest X-Ray Pneumonia Dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
![Chest Xray Dataset](https://github.com/user-attachments/assets/00d6e5a8-928e-4dc5-8169-0e17d9cb72c2)

  - **Brain Tumor:** [Brain Tumor Classification MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
![Brain Tumor Dataset](https://github.com/user-attachments/assets/6f89bf30-a3b2-4994-9e25-e9aa38dde4be)

## Resources

- [**Official Report**](https://github.com/guthlb/Comparitive-Analysis-of-Hybrid-ML-and-DL-models-for-Medical-Image-Classification/blob/main/Comparative%20Analysis%20of%20SVM%2C%20CNN%2C%20and%20XGBoost%20for%20Binary%20and%20Multiclass%20Classification.pdf)
  For an in-depth analysis of the project’s methodologies, experimental results, and comparative findings, refer to the official report. The document covers detailed discussions on standalone ML, DL, and hybrid approaches applied to medical image classification.

- [**Summary Presentation**](https://github.com/guthlb/Comparitive-Analysis-of-Hybrid-ML-and-DL-models-for-Medical-Image-Classification/blob/main/Summary%20Presentation.pdf) 
  For a concise overview of the project, including key workflows, architecture diagrams, and result highlights, check out the summary presentation slides. This is ideal for a quick understanding of the project’s objectives and outcomes.
## Requirements

- **Python 3.6 or higher**
- **pip (Python package installer)**
- **Required Libraries:**
  - TensorFlow and Keras (for deep learning model development)
  - scikit-learn (for traditional ML models and evaluation)
  - NumPy and PIL (for image processing)
  - Pandas, Matplotlib, Seaborn (for data handling and visualization)
- **GPU (reccomended for faster training):** Kaggle Notebook or Google Colab for GPU access.

## Setup Instructions

```bash
#1. Clone the repository
git clone https://github.com/guthlb/Comparitive-Analysis-of-Hybrid-ML-and-DL-models-for-Medical-Image-Classification
cd Comparitive-Analysis-of-Hybrid-ML-and-DL-models-for-Medical-Image-Classification

#2. If you do not have Python 3.6 or higher, create a virtual environment using this in your command prompt

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
# Windows
python -m venv venv
venv\Scripts\activate

#3. Download all the required packages by running:
pip install -r requirements.txt

#4. Open and run the desired .ipynb file in Jupyter Notebook, or use:
jupyter notebook

# For GPU acceleration, use Kaggle Notebook or Google Colab by uploading the code and data.
