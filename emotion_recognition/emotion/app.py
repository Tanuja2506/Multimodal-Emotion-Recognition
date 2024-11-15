import os
import mysql.connector
import flask_bcrypt
from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoConfig
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64


app = Flask(__name__, static_url_path='/static')
app.secret_key = 'your_secret_key'
bcrypt = Bcrypt(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

device = torch.device('cpu')
mtcnn = MTCNN(min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, device=device)
extractor = AutoFeatureExtractor.from_pretrained("trpakov/vit-face-expression")
model = AutoModelForImageClassification.from_pretrained("trpakov/vit-face-expression")
config = AutoConfig.from_pretrained("trpakov/vit-face-expression")
id2label = config.id2label

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}

def detect_emotions(image):
    try:
        temporary = image.copy()
        sample = mtcnn.detect(temporary)
        if sample[0] is not None:
            box = sample[0][0]
            face = temporary.crop(box)
            inputs = extractor(images=face, return_tensors="pt")
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probabilities = probabilities.detach().numpy().tolist()[0]
            class_probabilities = {id2label[i]: prob for i, prob in enumerate(probabilities)}
            return face, class_probabilities
    except Exception as e:
        print(f"Error in detect_emotions: {e}")
    return None, None

def process_video(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_class_probabilities = []

        for _ in range(0, frame_count, 5):  # Process every 5th frame
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            _, class_probabilities = detect_emotions(frame_pil)
            if class_probabilities:
                all_class_probabilities.append(class_probabilities)

        cap.release()

        if all_class_probabilities:
            df = pd.DataFrame(all_class_probabilities)
            overall_percentages = df.mean() * 100
            return all_class_probabilities, overall_percentages.to_dict()
        else:
            return None, None
    except Exception as e:
        print(f"Error processing video: {e}")
        return None, None

def create_emotion_plot(all_class_probabilities):
    try:
        df = pd.DataFrame(all_class_probabilities)
        df = df * 100
        plt.figure(figsize=(15, 8))
        colors = {
            "angry": "red", "disgust": "green", "fear": "gray",
            "happy": "yellow", "neutral": "purple", "sad": "blue", "surprise": "orange"
        }
        for emotion in df.columns:
            plt.plot(df[emotion], label=emotion, color=colors[emotion])
        plt.xlabel('Frame Order')
        plt.ylabel('Emotion Probability (%)')
        plt.title('Emotion Probabilities Over Time')
        plt.legend()

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_str = base64.b64encode(img_buf.getvalue()).decode()
        plt.close()
        return img_str
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None

@app.route('/index', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            all_class_probabilities, overall_percentages = process_video(filename)
            os.remove(filename)  # Clean-up after processing

            if all_class_probabilities:
                plot_img = create_emotion_plot(all_class_probabilities)
                return render_template('result.html', plot_img=plot_img, overall_percentages=overall_percentages)
            else:
                flash('No faces detected in the video')
                return redirect(request.url)
    return render_template('upload.html')

@app.route('/')
def home():
    return redirect('/index')

if __name__ == '__main__':
    app.run(debug=True)
