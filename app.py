from flask import Flask, render_template, request, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from datetime import datetime

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("model/pneumonia_model.h5")

UPLOAD_FOLDER = "static/uploads"
REPORT_FOLDER = "reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

# Preprocess image
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((150,150))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)
    return image

# Home
@app.route('/')
def home():
    return render_template("index.html")

# Predict
@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    age = request.form['age']

    file = request.files['image']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = Image.open(filepath)
    processed = preprocess_image(img)

    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        result = "Pneumonia Detected"
    else:
        result = "Normal"

    confidence = round(float(prediction) * 100, 2)

    return render_template(
        "result.html",
        result=result,
        confidence=confidence,
        image=filepath,
        name=name,
        age=age
    )

# Download Report
@app.route('/download_report', methods=['POST'])
def download_report():
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    name = request.form['name']
    age = request.form['age']
    result = request.form['result']
    confidence = request.form['confidence']

    filename = f"{name}_report.pdf"
    filepath = os.path.join(REPORT_FOLDER, filename)

    doc = SimpleDocTemplate(filepath)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AI Pneumonia Detection Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Patient Name: {name}", styles['Normal']))
    content.append(Paragraph(f"Age: {age}", styles['Normal']))
    content.append(Paragraph(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))

    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Result: {result}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))

    content.append(Spacer(1, 20))

    content.append(Paragraph(
        "Note: This is an AI-generated report. Please consult a doctor for confirmation.",
        styles['Italic']
    ))

    doc.build(content)

    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)