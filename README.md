# 🧠 MNIST Digit Classifier (Detecting Digit "5")

This project implements a simple **binary classifier** using the **SGD (Stochastic Gradient Descent) algorithm** from Scikit-Learn to identify whether a given handwritten digit (from the MNIST dataset) is **the digit 5** or not.

It demonstrates the complete workflow of a machine learning pipeline — from data loading and preprocessing to model training, prediction, and evaluation.

---

## 📂 Project Overview
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels.  
In this project, the task is simplified to **binary classification**:  
> “Is the digit a 5 or not?”

The model is trained using **Scikit-Learn’s `SGDClassifier`**, a linear model that is fast and effective for large-scale datasets.

---

## ⚙️ Workflow

1. **Data Loading:**  
   Loaded the MNIST dataset from OpenML using `fetch_openml("mnist_784")`.

2. **Preprocessing:**  
   - Converted the labels to unsigned integers (`uint8`).  
   - Created binary labels (`True` for digit 5, `False` otherwise`).  
   - Split the data into training (60,000) and test (10,000) sets.  

3. **Model Training:**  
   - Trained an `SGDClassifier` on the training data.  
   - Used default hyperparameters for simplicity.  

4. **Evaluation:**  
   - Measured accuracy on both training and test sets using `accuracy_score`.  
   - Displayed sample digits using Matplotlib.  

---

## 📊 Results
- **Training Accuracy:** ~100%  
- **Test Accuracy:** typically around **90–92%**, depending on random initialization and environment.  

The model successfully distinguishes digit "5" from other digits in most cases.

---

## 🧩 Technologies Used
- Python  
- Scikit-Learn  
- NumPy  
- Matplotlib  

---

## 🚀 How to Run

```bash
# Clone this repository
git clone https://github.com/<your-username>/mnist-classifier.git
cd mnist-classifier

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook classifier.ipynb
```

---

## 📦 requirements.txt
```txt
scikit-learn
numpy
matplotlib
```

---

  
