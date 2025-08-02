from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from datetime import datetime
import mysql.connector
import imutils

# ------------------- App Setup -------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_PATH = os.path.join('models', 'cnn-parameters-improvement-02-0.96.keras')
model = load_model(MODEL_PATH)

AUTHORIZED_USER = {
    'email': 'doctor@gmail.com',
    'password': 'drpass123'
}

# ------------------- Helper Functions -------------------

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="kidney_db"
    )

def crop_kidney_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=2)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    if not contours:
        raise ValueError("No kidney contour found in the image.")
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return image[y:y+h, x:x+w]

def predict_cancer(filepath):
    if not is_allowed_file(filepath):
        raise ValueError("Unsupported file type.")
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError("Image failed to load.")
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Invalid format. MRI must be RGB.")
    b, g, r = cv2.split(image)
    if not (np.allclose(b, g, atol=5) and np.allclose(g, r, atol=5)):
        raise ValueError("Upload grayscale MRI only.")
    brightness = np.mean(image)
    if brightness < 15 or brightness > 245:
        raise ValueError("Image brightness not acceptable.")
    if image.shape[0] < 100 or image.shape[1] < 100:
        raise ValueError("Image resolution too low.")
    try:
        image = crop_kidney_contour(image)
    except Exception as e:
        raise ValueError(f"Contour error: {str(e)}")
    image = cv2.resize(image, (240, 240))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0][0]
    print(f"[DEBUG] Prediction raw: {prediction:.4f}")
    return f"Cancer Detected (Confidence: {prediction:.2f})" if prediction >= 0.5 else f"Normal (Confidence: {1 - prediction:.2f})"

# ------------------- Routes -------------------

# Home page
@app.route('/')
def home():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

# About page
@app.route('/about')
def about():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    return render_template('about.html', modality="MRI Only")

# Prediction page
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    result, error = None, None
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        image_file = request.files.get('image')
        if not image_file or image_file.filename == '':
            error = "Please upload an MRI image."
        elif not is_allowed_file(image_file.filename):
            error = "File type not allowed."
        else:
            filename = secure_filename(image_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], datetime.now().strftime("%Y%m%d%H%M%S_") + filename)
            image_file.save(filepath)
            try:
                result = predict_cancer(filepath)
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
                error = f"Prediction error: {str(e)}"
    return render_template('prediction.html', result=result, error=error)

# Report page
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

# Analysis page
@app.route('/analysis')
def analysis():
    if 'user_email' not in session:
        return redirect(url_for('login'))
    db = connect_db()
    cursor = db.cursor()
    cursor.execute("SELECT COUNT(*) FROM reports")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM reports WHERE result LIKE '%Normal%'")
    normal = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM reports WHERE result LIKE '%Cancer%'")
    abnormal = cursor.fetchone()[0]
    cursor.execute("SELECT DATE(date) as report_date, COUNT(*) FROM reports GROUP BY DATE(date)")
    rows = cursor.fetchall()
    labels = [row[0].strftime('%Y-%m-%d') for row in rows]
    values = [row[1] for row in rows]
    cursor.close()
    db.close()
    return render_template('analysis.html',
                           total=total, normal=normal, abnormal=abnormal,
                           labels=labels, values=values)

# Login page
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

# Logout route
@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('login'))

# ------------------- Run App -------------------
if __name__ == '__main__':
    app.run(debug=True)
