from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from datetime import datetime
import mysql.connector
import imutils

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

AUTHORIZED_USER = {
    'email': 'doctor@gmail.com',
    'password': 'drpass123'
}

MODEL_PATH = os.path.join('models', 'cnn-parameters-improvement-02-0.96.keras')
model = load_model(MODEL_PATH)

# ---------- Helper Functions ----------

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
        raise ValueError("Unsupported file type. Use .png, .jpg, or .jpeg only.")

    image = cv2.imread(filepath)
    if image is None:
        raise ValueError("Image failed to load. Ensure it's a valid MRI.")

    # 🔒 Image quality checks before crop
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_mean = np.mean(hsv[:, :, 1])
    brightness = np.mean(image)

    if sat_mean > 40:
        raise ValueError("Rejected: image too colorful. Upload grayscale MRI only.")
    if brightness < 15 or brightness > 245:
        raise ValueError("Rejected: image too dark or bright. Upload a clear MRI scan.")

    # 🧠 Cropping and prediction
    image = crop_kidney_contour(image)
    image = cv2.resize(image, (240, 240))
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]
    print(f"[DEBUG] Prediction raw: {prediction:.4f}")

    if prediction >= 0.5:
        return f"Cancer Detected (Confidence: {prediction:.2f})"
    else:
        return f"Normal (Confidence: {1 - prediction:.2f})"

# ---------- Routes ----------

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

    result, error = None, None

    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        gender = request.form.get('gender')
        image_file = request.files.get('image')

        if not image_file or image_file.filename == '':
            error = "Please upload a valid MRI image."
        elif not is_allowed_file(image_file.filename):
            error = "Unsupported file type. Use PNG, JPG, or JPEG only."
        else:
            base_filename = secure_filename(image_file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{base_filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(filepath)

            try:
                result = predict_cancer(filepath)
                print(f"[DEBUG] Final Prediction: {result}")

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
                print(f"[ERROR] {error}")

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

if __name__ == '__main__':
    app.run(debug=True)
