import streamlit as st
import joblib as jb
import numpy as np

# Step 1: Loading model and scaler
model = jb.load("breast_cancer_model.pkl")
scaler = jb.load("scaler.pkl")

# Step 2: Most important features only for as a Beginner
top_features = [
    "mean radius", "mean perimeter", "mean area", "mean concavity", "mean concave points",
    "worst radius", "worst perimeter", "worst area", "worst concavity", "worst concave points"
]

# Step 3: UI Setup using Stremlit
st.set_page_config(page_title="Breast Cancer Diagnosis App", layout="wide")
st.title("🔬 Breast Cancer Diagnosis App")
st.markdown("Enter the following **10 key medical features** to predict if the tumor is **Benign** or **Malignant**.")

# Step 4: Creating input fields
user_input = []
for feature in top_features:
    value = st.number_input(f"{feature}", min_value=0.0, format="%.3f")
    user_input.append(value)

# Step 5: Predicting when button is clicked
if st.button("🔍 Predict"):
    input_array = np.array(user_input).reshape(1, -1)

    if np.all(input_array == 0):
        st.error("⚠️ Please enter valid values before predicting.")
    else:
        # Step 6: Padding the remaining 20 features with zeros to match important input features
        input_array_padded = np.pad(input_array, ((0, 0), (0, 30 - input_array.shape[1])), 'constant')

        # Step 7: Scaling and predicting
        input_scaled = scaler.transform(input_array_padded)
        prediction = model.predict(input_scaled)[0]

        result = "🟢 Benign (Non-Cancerous)" if prediction == 1 else "🔴 Malignant (Cancerous)"
        st.subheader("🧪 Prediction Result:")
        st.success(result)


