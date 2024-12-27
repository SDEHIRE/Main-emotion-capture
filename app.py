import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import io
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image
from pymongo import MongoClient
import os

app = Flask(__name__)
CORS(app)

# MongoDB Configuration
MONGO_URI = "mongodb+srv://sdehire:1111@cluster0.pft5g.mongodb.net/"
DB_NAME = "sdehire"
COLLECTION_NAME = "codeanalysis"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# Set the allowed file extensions
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to handle image uploads and emotion analysis
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'message': 'No file part'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        if file and allowed_file(file.filename):
            # Read the image in memory using PIL and convert to OpenCV format
            image_data = file.read()
            img = Image.open(io.BytesIO(image_data))

            # Convert to OpenCV format (DeepFace expects RGB images)
            img_rgb = np.array(img.convert('RGB'))

            # Perform emotion analysis using the image in RGB format with enforce_detection=False
            try:
                # Pass the image to DeepFace for analysis
                analysis = DeepFace.analyze(img_rgb, actions=['emotion'], enforce_detection=False)

                # Log the analysis result for debugging
                print("DeepFace analysis result:", analysis)

                if not analysis or len(analysis) == 0:
                    raise ValueError("No emotion analysis results found.")

                # Extracting the emotion with the highest confidence
                dominant_emotion = analysis['dominant_emotion']
                confidence = analysis['emotion'][dominant_emotion]

                # All detected emotions and their corresponding confidence scores
                all_emotions = analysis['emotion']

                # Prepare the data to be stored in MongoDB
                analysis_data = {
                    "dominant_emotion": dominant_emotion,
                    "confidence": confidence,
                    "all_emotions": all_emotions
                }

                # Insert the analysis result into MongoDB
                collection.insert_one(analysis_data)

                return jsonify({
                    'message': 'File uploaded and emotion analysis successful',
                    'dominant_emotion': dominant_emotion,
                    'confidence': confidence,
                    'all_emotions': all_emotions
                }), 200

            except Exception as e:
                error_message = f"Error during emotion analysis: {str(e)}"
                print(error_message)
                return jsonify({'message': error_message}), 500

    except Exception as e:
        error_message = f"General error: {str(e)}"
        print(error_message)
        return jsonify({'message': error_message}), 500


if __name__ == '__main__':
    app.run(debug=True)
