# Breast Cancer Diagnosis Classifier using Logistic Regression

This project applies various forms of logistic regression to classify tumors as benign or malignant using the Breast Cancer Wisconsin Diagnostic dataset. It explores how different gradient descent strategies and L2 regularization impact model convergence and performance.

## 📌 Problem Statement
Given a dataset of tumor features, build models to classify tumors as **benign** or **malignant** using logistic regression and its variants.

## What I Did
- Preprocessed data: removed irrelevant columns, encoded labels, and normalized features.
- Implemented logistic regression using:
  - Batch Gradient Descent (BGD)
  - Stochastic Gradient Descent (SGD)
  - Mini-Batch Gradient Descent (MBGD)
  - Each with and without L2 regularization
- Visualized convergence using loss curves.
- Evaluated models on training and test sets using accuracy metrics.

## 🛠️ Tools & Technologies
- **Languages/Libraries:** Python, NumPy, Pandas, Scikit-learn, Matplotlib
- **ML Techniques:** Logistic Regression, L2 Regularization, Gradient Descent (BGD, SGD, MBGD), Evaluation Metrics

## 📁 Dataset
**Breast Cancer Wisconsin (Diagnostic) Dataset**  
📎 Source: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

## 🎓 Learning Outcomes

This project was developed as part of my machine learning coursework during my Master’s in Computer Science. Through this hands-on experience, I strengthened my understanding of:

- Implementing logistic regression models from scratch
- Exploring optimization methods like BGD, SGD, and MBGD
- Applying L2 regularization to improve generalization
- Evaluating and visualizing model performance through loss curves and accuracy metrics

This project helped bridge theoretical ML concepts with real-world application and model tuning.

## How to Run

```bash
git clone https://github.com/seninfarheen/breast-cancer-logistic-regression.git
cd breast-cancer-logistic-regression
pip install -r requirements.txt
# Then open and run the Jupyter notebook:

_ "Open `Breast_Cancer_LogisticRegression.ipynb` to explore the implementation."_

- Consider renaming your `.ipynb` file to something readable and matching the project.

