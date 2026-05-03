🧠 Skin Lesion Detection using Deep Learning

📌 Overview

This project focuses on detecting and classifying skin lesions using deep learning techniques. The goal is to assist in early identification of potential skin diseases such as melanoma by analyzing dermoscopic images.

The model is trained on labeled image data and can predict lesion categories from new input images.

---

🚀 Features

- 🧬 Deep learning-based skin lesion classification
- 📊 Data preprocessing and validation pipeline
- ⚠️ Invalid image detection and handling
- 🔥 Grad-CAM visualization for model interpretability
- 📈 Training and evaluation scripts included

---

🛠️ Tech Stack

- Python
- TensorFlow / PyTorch
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn

---

📂 Project Structure

skin-lesion-detection/
│── app.py                         # Main application
│── model_utils.py                 # Model helper functions
│── train_melanoma_with_invalid_detection.py  # Training script
│── validate_dataset.py            # Dataset validation
│── gradcam.py                     # Visualization (Grad-CAM)
│── predict_test.py                # Testing predictions
│── requirements.txt               # Dependencies
│── templates/                     # UI (if applicable)
│── dataset/                       # (ignored)
│── models/                        # (ignored)
│── logs/                          # (ignored)

---

⚙️ Installation

1️⃣ Clone the repository

git clone https://github.com/krishnapp2004/skin-lesion-detection.git

2️⃣ Create virtual environment

python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies

pip install -r requirements.txt

---

📊 Dataset

The dataset is not included in this repository due to size limitations.

You can use publicly available datasets such as:

- ISIC (International Skin Imaging Collaboration)

Place your dataset inside:

dataset/
   ├── train/
   └── test/

---

▶️ How to Run

🔹 Train the model

python train_melanoma_with_invalid_detection.py

🔹 Validate dataset

python validate_dataset.py

🔹 Run prediction

python predict_test.py

---

🔥 Grad-CAM Visualization

Grad-CAM is used to visualize which parts of the image influenced the model’s prediction.

Run:

python gradcam.py

---

📈 Results

- Model trained on skin lesion dataset
- Achieves good classification performance (update with accuracy if available)
- Supports visualization for better interpretability

---

⚠️ Notes

- Dataset and trained models are excluded using ".gitignore"
- Make sure to download dataset separately
- Large files are not pushed to GitHub

---

💡 Future Improvements

- Improve model accuracy with better architectures
- Deploy as a web application
- Add real-time image prediction
- Integrate with mobile applications

---

🙌 Author

P P Krishna
Computer Science Student

---

⭐ If you like this project

Give it a star ⭐ on GitHub!
