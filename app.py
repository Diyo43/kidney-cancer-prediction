from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import imutils
import os
from datetime import datetime
import mysql.connector

# --- Database Connection ---
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="kidney_db"
    )

# --- Load Trained CT Model ---
model = load_model('models/cnn-parameters-improvement-03-1.00.keras')

# --- Authorized Doctor Credentials ---
AUTHORIZED_USER = {
    'email': 'doctor@gmail.com',
    'password': 'drpass123'
}

# --- Validate Kidney Image Format ---
def is_valid_kidney_image(image):
    if image is None:
        return False
    h, w = image.shape[:2]
    if h < 100 or w < 100:
        return False
    if len(image.shape) == 3 and np.mean(np.abs(image[..., 0] - image[..., 1])) > 20:
        return False
    return True

# --- Cancer Prediction Function for CT ---
def predict_cancer(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise ValueError("Failed to load image.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        raise ValueError("No contours found in image.")

    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    image = cv2.resize(image, (240, 240))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]
    label = "Cancer Detected" if prediction > 0.5 else "Normal"
    return label, float(prediction)

# --- Flask App Config ---
app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = r'C:\Users\pc\Music\Kidney cancer detection\Dataset'

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
    return render_template('about.html', modality="CT and MRI Support Coming Soon")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_email' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        scan_type = request.form.get('scan_type')
        image_file = request.files['image']

        # Block MRI support for now
        if scan_type == "MRI":
            return render_template("prediction.html", error="MRI scan support is coming soon. Please upload a CT scan.")

        # Validate file
        if image_file.filename == '':
            return render_template("prediction.html", error="No image selected.")

        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if os.path.exists(filepath):
            os.remove(filepath)
        image_file.save(filepath)

        image = cv2.imread(filepath)
        if not is_valid_kidney_image(image):
            os.remove(filepath)
            return render_template('prediction.html', error="Invalid image. Please upload a real kidney CT scan.")

        result, score = predict_cancer(filepath)
        date = datetime.now().strftime("%Y-%m-%d %H:%M")

        db = connect_db()
        cursor = db.cursor()
        sql = "INSERT INTO reports (name, age, gender, scan_type, result, date) VALUES (%s, %s, %s, %s, %s, %s)"
        values = (name, age, gender, scan_type, result, date)
        cursor.execute(sql, values)
        db.commit()
        cursor.close()
        db.close()

        return redirect(url_for('report'))

    return render_template('prediction.html')

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
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        if email == AUTHORIZED_USER['email'] and password == AUTHORIZED_USER['password']:
            session['user_email'] = email
            return redirect(url_for('home'))
        return render_template('login.html', error="Access denied: Doctor only")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('login'))

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)
