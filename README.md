# Fraud Detection using Bayesian Statistics

This project explores fraud detection in financial transactions using **Bayesian statistics**, with a focus on **Gaussian Naïve Bayes** modeling and **deep learning** techniques. It was conducted as part of an academic initiative under the supervision of **Dr. Amr Amin** and **Eng. Omnia**.

## 📊 Dataset

We used a synthetically generated dataset from [Kaggle](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection/data), containing **5 million financial transactions** designed to simulate real-world fraud detection scenarios.

Each transaction includes:
- **Transaction Details**: ID, timestamp, sender/receiver, amount, transaction type
- **Behavioral Features**: Time since last transaction, spending deviation, velocity score, geo-anomaly
- **Metadata**: Location, device, channel, IP, hash
- **Fraud Indicators**: Binary fraud label (`is_fraud`), type of fraud (e.g., money laundering)

---

## 🔧 Steps & Workflow

### 1. Data Preprocessing

- **Feature Selection**: Removed ID, timestamps, and `fraud_type` columns.
- **Handling Missing Data**: Missing values in `time_since_last_transaction` were replaced with `-1`.
- **Handling Imbalanced Data**:
  - Fraud cases: ~180K
  - Non-fraud cases: ~4.82M
  - Used **downsampling** to reduce imbalance and help models generalize better.
- **Encoding and Standardization**: 
  - Encoded categorical features.
  - Standardized numerical features.

---

### 2. Hyperparameter Optimization using Optuna

Due to the highly **imbalanced dataset**, we did **not assume balanced class distributions**. Instead, we focused on models that can **adapt to imbalance** with the help of hyperparameter tuning.

- Used **Optuna** for tuning:
  - Based on **Tree-Structured Parzen Estimator (TPE)**.
  - Learns from past trials to suggest better parameters.
  - Includes **early stopping** via pruning.
- Techniques:
  - **Bayesian Optimization**
  - **Kernel Density Estimation (KDE)**

Downsampling was applied **before** training to assist models in better learning fraud patterns.

---

### 3. Model Building

#### A. Gaussian Naïve Bayes (GNB)

- **Purpose**: Baseline model using Bayes’ Theorem.
- **Assumption**: Features are normally distributed and independent.
- **Benefits**:
  - Lightweight and fast
  - Surprisingly effective for small/balanced data
- **Approach**: Applied **after downsampling** to reduce class imbalance.

#### B. Deep Learning (Dense Neural Network)

- **Architecture**:
  - Input Layer: Based on feature count
  - Hidden Layers: Dense layers with ReLU
  - Output Layer: Sigmoid for binary classification
- **Training**:
  - Used Binary Cross Entropy
  - Applied Dropout to prevent overfitting
- **Advantages**:
  - Learns complex patterns
  - Scales to large datasets

---

## 📈 Results

- **Gaussian Naïve Bayes** provided a solid and interpretable baseline.
- **Deep Neural Networks** outperformed on the full dataset after tuning and regularization.

---

## 🧠 Conclusion

The project highlights the importance of adapting models to **imbalanced datasets**, not just relying on assumptions of balance. By combining **Bayesian learning**, **Optuna optimization**, and **sampling techniques**, we built effective models for real-world fraud detection.

---

## 👥 Team

- **Taha Fawzy** 
- **Mostafa Mohamed** 
- Supervised by **Dr. Amr Amin**  
- Guided by **Eng. Omnia**
