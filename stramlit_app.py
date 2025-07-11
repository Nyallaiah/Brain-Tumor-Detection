import streamlit as st
import requests
from PIL import Image

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI or CT scan to check for brain tumors.")

uploaded_file = st.file_uploader("Upload an MRI or CT Scan", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/predict", files=files)

        if response.status_code == 200:
            result = response.json()
            st.write(f"### **Tumor Type:** {result['tumor_type']}")
        else:
            st.write("Error in prediction.")

