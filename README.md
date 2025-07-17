# 🔬 Breast Cancer Diagnosis App

This is a machine learning-powered web app built with **Scikit-learn** and **Streamlit** that predicts whether a tumor is **Benign or Malignant** using key medical features.

## 🚀 Features

- Predicts tumor diagnosis in real-time
- Takes top 10 predictive input features
- Built using RandomForestClassifier (or your model)
- Clean, interactive Streamlit UI
  
## 📝 Requirements

- scikit-learn
- streamlit
- numpy
- joblib
  
## 📦 Files

- `app.py`: Main Streamlit app
- `breast_cancer_model.pkl`: Saved trained model
- `scaler.pkl`: StandardScaler object for preprocessing

## 💻 How to Run

```bash
streamlit run app.py

