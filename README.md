# Skin Lesion Detection Using Deep Learning - SKINAI

A deep learning-based web application for classifying dermoscopic skin lesion images into nine categories, including melanoma. The system uses a lightweight MobileNetV2 architecture for efficient image classification and integrates Grad-CAM for model interpretability.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic system and should not be used as a substitute for professional medical advice or clinical examination.

## Overview

Skin cancer is one of the most common forms of cancer, and early identification of suspicious lesions can play an important role in improving treatment outcomes. This project explores the application of deep learning and computer vision for automated skin lesion classification.

The system was developed using **MobileNetV2 with transfer learning** and trained on **27,666 dermoscopic images** across nine lesion categories. The resulting model achieved **85.3% overall classification accuracy** and **87.2% sensitivity for melanoma detection**.

The trained model is deployed through a Flask-based web application that allows users to upload an image, obtain a prediction, view the model's confidence, and examine a Grad-CAM visualization of the regions influencing the prediction.

## Key Features

* Nine-class skin lesion classification
* Melanoma detection
* MobileNetV2-based transfer learning
* Grad-CAM-based model explainability
* Real-time image prediction
* Lightweight 13.4 MB trained model
* Prediction latency of less than three seconds
* Web-based image upload and prediction interface
* Educational module covering the ABCDE rule
* Skin cancer awareness and prevention information
* Invalid or unsuitable image detection

## Model Performance

The model was trained using 27,666 dermoscopic images.

| Metric               | Performance |
| -------------------- | ----------: |
| Overall Accuracy     |       85.3% |
| Melanoma Sensitivity |       87.2% |
| Number of Classes    |           9 |
| Training Images      |      27,666 |
| Model Size           |     13.4 MB |
| Prediction Time      | < 3 seconds |
| Architecture         | MobileNetV2 |

Melanoma sensitivity was specifically considered as an important evaluation metric because minimizing false-negative predictions is particularly relevant for melanoma screening applications.

## Model Architecture

The project uses **MobileNetV2**, a lightweight convolutional neural network architecture designed for efficient image classification.

Transfer learning is used to leverage features learned from large-scale image datasets while adapting the model to the skin lesion classification task.

### Prediction Pipeline

```text
Input Image
     |
     v
Image Preprocessing
     |
     v
MobileNetV2
     |
     v
Classification Layer
     |
     v
Nine-Class Prediction
     |
     +------------------+
     |                  |
     v                  v
Confidence Score    Grad-CAM
                        |
                        v
                 Visual Explanation
```

The lightweight architecture enables the model to perform predictions efficiently while maintaining competitive classification performance.

## Explainable AI

To improve model interpretability, the application incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)**.

Grad-CAM generates a heatmap indicating the regions of the input image that contributed most strongly to the model's prediction. This provides a visual representation of the model's attention and helps users better understand the basis of its classification.

The explainability component is particularly useful when working with deep learning models, where individual predictions can otherwise be difficult to interpret.

## Web Application

The trained model is integrated into a **Flask web application** that provides a simple interface for interacting with the system.

### Application Workflow

1. Upload a dermoscopic image.
2. Preprocess the uploaded image.
3. Run the image through the trained MobileNetV2 model.
4. Generate the predicted lesion category.
5. Display the prediction confidence.
6. Generate a Grad-CAM visualization.
7. Present relevant skin cancer awareness information.

## Educational Module

The application includes an educational section designed to provide general information about skin cancer awareness.

The module introduces the **ABCDE rule**:

| Criterion     | Description                                                          |
| ------------- | -------------------------------------------------------------------- |
| A — Asymmetry | One half of a lesion differs from the other half.                    |
| B — Border    | The border may be irregular, uneven, or poorly defined.              |
| C — Color     | The lesion may contain multiple or unevenly distributed colors.      |
| D — Diameter  | Larger lesions may require additional clinical attention.            |
| E — Evolving  | Changes in size, shape, color, or appearance may require evaluation. |

The application also provides general information regarding skin cancer awareness and prevention.

## Project Structure

```text
skin-lesion-detection-miniproject/
│
├── Outputs/
│
├── app.py
├── model_utils.py
├── train_melanoma_with_invalid_detection.py
├── requirements.txt
├── README.md
└── .gitignore
```

### File Description

| File                                       | Description                            |
| ------------------------------------------ | -------------------------------------- |
| `app.py`                                   | Flask application and web interface    |
| `model_utils.py`                           | Model loading and prediction utilities |
| `train_melanoma_with_invalid_detection.py` | Model training and evaluation pipeline |
| `requirements.txt`                         | Python package dependencies            |
| `Outputs/`                                 | Generated results and outputs          |
| `README.md`                                | Project documentation                  |

## Technology Stack

**Programming Language**

* Python

**Deep Learning**

* TensorFlow
* Keras
* MobileNetV2

**Computer Vision and Data Processing**

* OpenCV
* NumPy
* Matplotlib

**Explainable AI**

* Grad-CAM

**Web Development**

* Flask
* HTML
* CSS

**Development Tools**

* Visual Studio Code
* Jupyter Notebook
* Git
* GitHub

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/krishnapp2004/skin-lesion-detection-miniproject.git
cd skin-lesion-detection-miniproject
```

### 2. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Flask application using:

```bash
python app.py
```

The application will start on the local Flask server. Open the URL displayed in the terminal to access the web interface.

## Model Training

The model can be trained using the provided training script:

```bash
python train_melanoma_with_invalid_detection.py
```

The training pipeline includes image preprocessing, model training, validation, and evaluation.

The complete dataset and trained model files may not be included in the repository due to their size.

## Dataset

The model was trained using **27,666 dermoscopic images** representing nine skin lesion categories.

The dataset is not included directly in this repository because of its size and dataset distribution considerations. To reproduce the project, the dataset should be obtained from its appropriate source and organized according to the structure expected by the training pipeline.

## Results

The trained MobileNetV2 model achieved:

* **85.3% overall accuracy**
* **87.2% melanoma sensitivity**
* **13.4 MB model size**
* **Less than 3 seconds prediction time**

These results demonstrate the feasibility of deploying a lightweight deep learning model for real-time skin lesion classification through a web-based interface.

## Limitations

The system has several limitations:

* Model performance depends on the quality and distribution of the training data.
* Dermoscopic images may differ significantly from images captured using consumer cameras or smartphones.
* The model can produce incorrect predictions.
* Performance may vary across different populations, imaging devices, and clinical environments.
* The system has not been validated for clinical diagnosis.
* Predictions should not be considered a substitute for evaluation by a qualified medical professional.

## Future Scope

Potential improvements include:

* Experimenting with architectures such as EfficientNet, ResNet, and Vision Transformers.
* Improving classification performance through advanced augmentation and optimization techniques.
* Incorporating lesion segmentation before classification.
* Expanding the dataset with more diverse images.
* Improving performance on real-world smartphone images.
* Deploying the application on a cloud platform.
* Developing a dedicated mobile application.
* Adding detailed per-class precision, recall, and F1-score analysis.
* Extending the explainability module with additional XAI techniques.
* Supporting multiple languages for the educational content.

## Project Impact

This project demonstrates an end-to-end application of deep learning, explainable AI, and web development to the problem of skin lesion classification.

By combining a lightweight classification model with Grad-CAM visualization and an accessible web interface, the system provides a practical prototype for AI-assisted skin lesion screening and skin cancer awareness.

The primary objective is to demonstrate the technical feasibility of using deep learning for image-based skin lesion analysis while emphasizing interpretability and responsible use.

## Author

**P. P. Krishna**

Computer Science and Engineering Student

GitHub: [krishnapp2004](https://github.com/krishnapp2004)

## License

This project is intended for educational and research purposes.
