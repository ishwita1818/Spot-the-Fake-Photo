# Spot-the-Fake-Photo

## Overview
This project detects whether an input image is a **real photograph** or a **recaptured image displayed on a screen** using handcrafted computer vision features and a calibrated machine learning model. The solution is lightweight, computationally efficient, and suitable for on-device inference.

## Dataset
A custom balanced dataset containing **120 images** was created:
- 60 Real Images
- 60 Screen Images

Images were captured under different viewing angles, lighting conditions, brightness levels, zoom levels, and distances to improve generalization.

## Methodology
- Image preprocessing and normalization
- Handcrafted feature extraction:
  - Global Laplacian Variance
  - Patch-wise Laplacian Statistics
  - FFT Frequency Ratios
  - Edge Density
  - Saturation Variance
  - Specular Highlight Count
- Feature scaling using **RobustScaler**
- Classification using **Logistic Regression**
- Probability calibration using **CalibratedClassifierCV**
- Threshold optimization for improved classification performance(Threshold of 0.471)

## Model Evaluation
The model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Stratified Cross Validation

## Repository
```
predict.py
best_model.joblib
scaler.joblib
SalescodeAi_Ishwita.ipynb
requirements.txt
```

## Usage
```bash
python predict.py image.jpg
```

The script returns a probability between **0 and 1**, where:
- **0** → Real Photograph
- **1** → Screen Recaptured Image

## Technologies
Python, OpenCV, NumPy, Scikit-learn, SciPy, Pillow, Pandas, Joblib

## Future Improvements
- Larger and more diverse dataset
- Additional frequency and texture descriptors
- Lightweight CNN-based feature extraction
- Improved robustness across different display technologies
