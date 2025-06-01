from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from datetime import datetime
import mysql.connector

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Database Connection ---
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="kidney_db"
    )

# --- Load Trained Model ---
model = load_model('models/cnn-parameters-improvement-03-1.00.keras')

# --- Hardcoded Doctor Login ---
AUTHORIZED_USER = {
    'email': 'doctor@gmail.com',
    'password': 'drpass123'
}

# --- Image Validation for MRI Scan ---
def is_valid_mri(image):
    if image is None or len(image.shape) < 2:
        return False
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return False
    # Accept grayscale or near-grayscale images
    if len(image.shape) == 2 or image.shape[2] == 1:
        return True
    b, g, r = cv2.split(image)
    return (np.mean(np.abs(r - g)) < 15 and
            np.mean(np.abs(r - b)) < 15 and
            np.mean(np.abs(g - b)) < 15)

# --- Prediction Logic ---
def predict_cancer(filepath):
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError("Invalid image loaded.")
    
    # Convert BGR (OpenCV default) to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize to (240, 240)
    image = cv2.resize(image, (240, 240))
    
    # Normalize pixel values to [0, 1]
    image = image / 255.0
    
    # Add batch dimension: (1, 240, 240, 3)
    image = np.expand_dims(image, axis=0)
    
    # Predict with model
    prediction = model.predict(image)[0][0]
    print(f"[DEBUG] Prediction: {prediction:.4f}")
    
    # Interpret prediction (adjust threshold if needed)
    if prediction > 0.5:
        return "Normal"
    else:
        return "Cancer Detected"

# --- Routes ---
@app.route('/')
def home():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/about')
def about():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('about.html', modality="MRI Only")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    result = None
    error = None

    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        image_file = request.files.get('image')

        if not image_file or image_file.filename == '':
            error = "Please upload a valid image."
        else:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Remove file if it already exists
            if os.path.exists(filepath):
                os.remove(filepath)

            image_file.save(filepath)
            image = cv2.imread(filepath)

            if not is_valid_mri(image):
                os.remove(filepath)
                error = "Invalid image. Please upload a grayscale-like MRI scan."
            else:
                try:
                    result = predict_cancer(filepath)

                    # Save result to database
                    db = connect_db()
                    cursor = db.cursor()
                    cursor.execute("""
                        INSERT INTO reports (name, age, gender, scan_type, result, date)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (name, age, gender, "MRI", result, datetime.now().strftime("%Y-%m-%d %H:%M")))
                    db.commit()
                    cursor.close()
                    db.close()
                except Exception as e:
                    error = f"Error during prediction or DB save: {str(e)}"

    return render_template('prediction.html', result=result, error=error)

@app.route('/report')
def report():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    db = connect_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("SELECT * FROM reports ORDER BY id DESC")
    reports = cursor.fetchall()
    cursor.close()
    db.close()
    return render_template('report.html', reports=reports)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == AUTHORIZED_USER['email'] and password == AUTHORIZED_USER['password']:
            session['user_email'] = email
            return redirect(url_for('home'))
        else:
            error = "Access denied: Doctor only."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('login'))

# --- Run Application ---
if __name__ == '__main__':
    app.run(debug=True)
