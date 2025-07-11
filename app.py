import os
import cv2
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import base64
from io import BytesIO
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import shap
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import json
from datetime import datetime
import csv
from fpdf import FPDF  # pip install fpdf
from flask import send_file
import zipfile
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, session, url_for, flash
from flask_mysqldb import MySQL
from flask_mysqldb import MySQL
import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator, array_to_img
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import shap
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
import json
from datetime import datetime
import csv
from fpdf import FPDF
import zipfile
import shutil
import pandas as pd
from flask_mysqldb import MySQL
import tensorflow as tf
import os
from flask import Flask
from datetime import datetime
from flask import Flask, request, jsonify
import openai

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)






# Add this ro

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'b86f504f28c47a422a314a535b3b7bb9'
openai.api_key = os.getenv("OPENAI_API_KEY")

# MySQL Configuration

app.config['MYSQL_HOST'] = 'localhost'  # ✅ or your actual server's IP
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Yallaiah@34'
app.config['MYSQL_DB'] = 'brain_tumor_db'

# Initialize MySQL
mysql = MySQL(app)

# Your other routes and logic...


# Load the trained model
MODEL_PATH = "CNN.h5"
model = load_model(MODEL_PATH)
PATIENT_RECORDS_PATH = 'static/data/patient_records.json'


IMG_HEIGHT = 150
IMG_WIDTH = 150

def predict_image(image_path):
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    label = "Tumor Detected" if prediction > 0.5 else "Healthy"
    return label, float(prediction)

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in the model.")

os.makedirs(os.path.dirname(PATIENT_RECORDS_PATH), exist_ok=True)
if not os.path.exists(PATIENT_RECORDS_PATH):
    with open(PATIENT_RECORDS_PATH, 'w') as f:
        json.dump([], f)

def load_patient_records():
    with open(PATIENT_RECORDS_PATH, 'r') as f:
        return json.load(f)

def save_patient_records(records):
    with open(PATIENT_RECORDS_PATH, 'w') as f:
        json.dump(records, f, indent=4)


def save_training_plots(history):
    # Plot and save loss
    plt.figure()
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('static/plots/loss_plot.png')
    plt.close()

    # Plot and save accuracy
    plt.figure()
    plt.plot(history['accuracy'], label='Train Acc')
    plt.plot(history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('static/plots/accuracy_plot.png')
    plt.close()


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return '<h2>About Brain Tumor Detection...</h2>'  # Replace with template if desired
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cur.fetchone()
        cur.close()
        if user:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash("Invalid credentials")
            return redirect(url_for('login'))
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
        mysql.connection.commit()
        cur.close()
        flash("Registration successful. Please login.")
        return redirect(url_for('login'))
    return render_template('register.html')



@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            return redirect(request.url)
        file = request.files['file']
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        label, confidence = predict_image(file_path)
        return render_template('index.html', uploaded_image=file.filename, label=label, confidence=confidence)

    return render_template('index.html', uploaded_image=None)


@app.route('/index')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', uploaded_image=None)


@app.route('/metrics')
def metrics():
    # Example static metrics (replace with actual ones if available)
    metrics_data = {
        "accuracy": 0.94,
        "precision": 0.91,
        "recall": 0.89,
        "f1_score": 0.90
    }
    return render_template('metrics.html', metrics=metrics_data)


@app.route('/grad', methods=['GET', 'POST'])
def grad_cam():
    grad_image = None
    original_image = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            original_image = file.filename

            # Generate dummy Grad-CAM path for now (replace with actual logic later)
            grad_image = 'grad_' + file.filename
            grad_image_path = os.path.join(app.config['UPLOAD_FOLDER'], grad_image)

            # You'd run your Grad-CAM function here and save to `grad_image_path`
            # For demo: copy uploaded file (simulate output)
            from shutil import copyfile
            copyfile(file_path, grad_image_path)

    return render_template('grad.html', grad_image=grad_image, original_image=original_image)


@app.route('/ensemble', methods=['GET', 'POST'])
def ensemble():
    uploaded_file = None
    prediction_details = {}

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            uploaded_file = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file)
            file.save(file_path)

            # Simulated predictions from three models
            model_preds = {
                'Model ANN': 0.82,
                'Model CNN': 0.75,
                'Model VGG16': 0.90
            }

            # Weighted ensemble example
            weights = {'Model ANN': 0.3, 'Model CNN': 0.3, 'Model VGG16': 0.4}
            final_score = sum(model_preds[m] * weights[m] for m in model_preds)
            label = "Tumor Detected" if final_score > 0.5 else "Healthy"

            prediction_details = {
                'uploaded_file': uploaded_file,
                'model_preds': model_preds,
                'final_score': round(final_score, 3),
                'label': label
            }

    return render_template('ensemble.html', **prediction_details)
@app.route('/other')
def other_features():
    return render_template('other.html')
@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    processed_img = None
    original_filename = None
    preprocessing_info = {}

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            original_filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(filepath)

            # Load image with OpenCV
            img = cv2.imread(filepath)
            original_shape = img.shape
            preprocessing_info['Original Dimensions'] = f"{original_shape[1]}x{original_shape[0]} (W x H)"

            # Resize
            resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            preprocessing_info['Resized to'] = f"{IMG_WIDTH}x{IMG_HEIGHT}"

            # Convert to grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            preprocessing_info['Color Mode'] = "Grayscale"

            # Histogram equalization
            equalized = cv2.equalizeHist(gray)
            preprocessing_info['Histogram Equalization'] = "Applied"

            # Normalize
            normalized = equalized / 255.0
            preprocessing_info['Normalization'] = "Pixel values scaled between 0 and 1"

            # Convert processed image to displayable format
            pil_img = Image.fromarray(equalized)
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            processed_img = base64.b64encode(buffer.getvalue()).decode()

    return render_template(
        'preprocess.html',
        original_filename=original_filename,
        processed_img=processed_img,
        preprocessing_info=preprocessing_info
    )

@app.route('/augment', methods=['GET', 'POST'])
def augment():
    augmented_images = []
    original_image = None

    if request.method == 'POST':
        file = request.files['file']
        selected_augmentations = request.form.getlist('augmentations')

        if file and file.filename:
            original_image = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_image)
            file.save(filepath)

            # Load image and preprocess
            image = load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
            x = img_to_array(image)
            x = np.expand_dims(x, axis=0)

            # Configure ImageDataGenerator dynamically
            datagen_args = {}

            if 'rotation' in selected_augmentations:
                datagen_args['rotation_range'] = 40
            if 'zoom' in selected_augmentations:
                datagen_args['zoom_range'] = 0.2
            if 'flip' in selected_augmentations:
                datagen_args['horizontal_flip'] = True
            if 'brightness' in selected_augmentations:
                datagen_args['brightness_range'] = [0.6, 1.4]

            datagen = ImageDataGenerator(**datagen_args)

            # Generate and convert to base64 for display
            i = 0
            for batch in datagen.flow(x, batch_size=1):
                img = array_to_img(batch[0])
                buffer = BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                augmented_images.append(img_str)
                i += 1
                if i >= 6:
                    break  # Limit to 6 images

    return render_template(
        'augment.html',
        original_image=original_image,
        augmented_images=augmented_images
    )

@app.route('/explain', methods=['GET', 'POST'])
def explain():
    original_filename = None
    explanation_img = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            original_filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(filepath)

            def predict_fn(images):
                images = np.array(images) / 255.0
                return model.predict(images)

            # Load and preprocess image
            img = load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img_array = img_to_array(img)
            img_array = np.uint8(img_array)

            # Create LIME explainer
            explainer = lime_image.LimeImageExplainer()
            explanation = explainer.explain_instance(
                img_array,
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=1000
            )

            # Get image with mask
            temp, mask = explanation.get_image_and_mask(
                label=explanation.top_labels[0],
                positive_only=True,
                hide_rest=False
            )

            lime_img = mark_boundaries(temp / 255.0, mask)
            lime_pil = Image.fromarray((lime_img * 255).astype(np.uint8))

            buffer = BytesIO()
            lime_pil.save(buffer, format="PNG")
            explanation_img = base64.b64encode(buffer.getvalue()).decode()

    return render_template(
        'explain.html',
        original_filename=original_filename,
        explanation_img=explanation_img
    )
@app.route('/records', methods=['GET', 'POST'])
def patient_records():
    records = load_patient_records()

    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        diagnosis = request.form['diagnosis']
        date = datetime.now().strftime("%Y-%m-%d %H:%M")

        new_record = {
            "name": name,
            "age": age,
            "diagnosis": diagnosis,
            "date": date
        }

        records.append(new_record)
        save_patient_records(records)

    query = request.args.get('query', '')
    if query:
        records = [r for r in records if query.lower() in r['name'].lower()]

    return render_template('record.html', records=records, query=query)
@app.route('/export', methods=['GET', 'POST'])
def export_reports():
    message = ""
    if request.method == 'POST':
        export_format = request.form.get('format')
        dummy_data = [
            {"name": "John Doe", "label": "Tumor Detected", "confidence": "0.87"},
            {"name": "Jane Smith", "label": "Healthy", "confidence": "0.12"}
        ]

        if export_format == 'csv':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'report.csv')
            with open(filepath, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=["name", "label", "confidence"])
                writer.writeheader()
                for row in dummy_data:
                    writer.writerow(row)
            message = "CSV report exported successfully."

        elif export_format == 'pdf':
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'report.pdf')
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Prediction Report", ln=True, align="C")
            pdf.ln(10)
            for row in dummy_data:
                pdf.cell(200, 10, txt=f"{row['name']} - {row['label']} - Confidence: {row['confidence']}", ln=True)
            pdf.output(filepath)
            message = "PDF report exported successfully."

    return render_template("export.html", message=message)

@app.route('/batch', methods=['GET', 'POST'])
def batch_predict():
    results = []
    download_path = None

    if request.method == 'POST':
        if 'zipfile' not in request.files or request.files['zipfile'].filename == '':
            return render_template('batch.html', results=results, download_path=None, error="No ZIP file uploaded")

        zip_file = request.files['zipfile']
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'batch_temp')
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.makedirs(temp_dir, exist_ok=True)

        zip_path = os.path.join(temp_dir, 'batch.zip')
        zip_file.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Process images
        for filename in os.listdir(temp_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(temp_dir, filename)
                label, confidence = predict_image(image_path)
                results.append({
                    "filename": filename,
                    "label": label,
                    "confidence": round(confidence, 3)
                })

        # Save results as CSV
        df = pd.DataFrame(results)
        download_path = os.path.join(app.config['UPLOAD_FOLDER'], 'batch_results.csv')
        df.to_csv(download_path, index=False)

    return render_template('batch.html', results=results, download_path=download_path, error=None)
@app.route('/metrics1')
def metrics1():


    return render_template('metrics1.html')

@app.route("/chatbot", methods=["POST"])
def chatbot_response():
    user_input = request.json['message']
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a compassionate medical assistant for a brain tumor detection app."},
            {"role": "user", "content": user_input}
        ]
    )
    return jsonify({"response": response['choices'][0]['message']['content']})


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
