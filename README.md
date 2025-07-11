🧠 Brain Tumor Detection Web Application
This is a comprehensive Flask-based web application for brain tumor detection using deep learning (CNN), powered with features like Grad-CAM, LIME explainability, ensemble predictions, image preprocessing/augmentation, patient record management, batch predictions, and report generation.

🚀 Features
🧪 Core Functionalities:
Brain Tumor Prediction (Binary classification: Tumor Detected / Healthy)

CNN Model trained on medical image data

Grad-CAM visualization for feature importance

LIME explanation to visualize decision areas

Ensemble Prediction from multiple models (Simulated)

Image Preprocessing with grayscale, resizing, normalization, histogram equalization

Data Augmentation (rotation, zoom, brightness, flipping)

🧾 Data & Records:
Patient records saved in JSON

Searchable patient records interface

Batch predictions via ZIP upload

Export results in CSV or PDF formats

💬 Chatbot Integration:
Integrated OpenAI GPT-3.5 chatbot as a virtual medical assistant

🧑‍💻 User Authentication:
User registration and login with session management

MySQL used for storing user credentials

🛠 Tech Stack
Frontend: HTML, CSS, Bootstrap, Jinja2

Backend: Python, Flask

Database: MySQL

AI Libraries: TensorFlow, Keras, OpenCV, LIME, SHAP

Visualization: Matplotlib

PDF/CSV Export: FPDF, CSV, Pandas

📁 Folder Structure
csharp
Copy
Edit
brain-tumor-app/
│
├── static/
│   ├── uploads/              # Uploaded images
│   ├── plots/                # Training accuracy/loss plots
│   └── data/patient_records.json
│
├── templates/
│   ├── home.html
│   ├── login.html
│   ├── register.html
│   ├── index.html
│   ├── metrics.html
│   ├── grad.html
│   ├── augment.html
│   ├── explain.html
│   ├── preprocess.html
│   ├── export.html
│   ├── batch.html
│   ├── record.html
│   └── metrics1.html
│
├── CNN.h5                   # Trained Keras CNN model
├── app.py                   # Main Flask application
└── requirements.txt         # Dependencies
📦 Setup Instructions
Clone the Repository

bash
Copy
Edit
git clone https://github.com/yourusername/brain-tumor-app.git
cd brain-tumor-app
Create Virtual Environment

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Requirements

bash
Copy
Edit
pip install -r requirements.txt
Configure MySQL

Create a MySQL database: brain_tumor_db

Create users table:

sql
Copy
Edit
CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(255),
  password VARCHAR(255)
);
Set OpenAI API Key

bash
Copy
Edit
export OPENAI_API_KEY="your-openai-key"
Run the Application

bash
Copy
Edit
python app.py
Open http://127.0.0.1:5000 in your browser.

✅ To Do / Improvements
Replace simulated ensemble model predictions with real models

Add model training pipeline

Integrate Grad-CAM with actual CNN layers

Add role-based login (doctor, patient)

Secure password handling (hashing)

Dockerize the app for deployment



🤝 Contributing
Pull requests and feature suggestions are welcome!
Open an issue to discuss what you want to improve.

📜 License
This project is licensed under the MIT License.
