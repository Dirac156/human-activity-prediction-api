import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, flash, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from collections import deque
from flask_cors import CORS

UPLOAD_FOLDER = 'static/uploads/'

# initialize the application
app = Flask(__name__)
CORS(app)
app.debug = 1
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

tf.config.run_functions_eagerly(True)

@app.route("/", methods=['GET'])
def index():
    """
    index: homepage
    render the homepage when a user visits the page
    """
    return "API is on", 200

@app.route("/upload", methods=['POST'])
def upload_video():
    """
    /upload: used to upload videos on the server
    returns a 200 http response when the video is upload
    returns a 400 http response when the video format isn't supported or when 
            size is greater than 10mb
    """
    if 'video' not in request.files:
        return jsonify({ 'message': 'No file uploaded'}), 400
    video = request.files['video']
    if video.filename == '':
        return jsonify({ 'message': 'No video selected for uploading'}), 400
    else:
        filename = secure_filename(video.filename)
        try:
            os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], "videos"))
        except:
            flash("folder exist")
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], "videos", filename))
        flash('Video successfully uploaded and displayed below')
    class_predicted = predict_on_live_video(os.path.join(app.config['UPLOAD_FOLDER'], "videos", filename))
    # delete video
    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], "videos", filename))
    return jsonify({ 'message': "prediciton complete", "value": class_predicted}), 200
    


def predict_on_live_video(video_file_path, window_size=25):
    classes_list = ["PlayingPiano", "PlayingViolin", "Drumming", "Basketball"]
    image_height, image_width = 64, 64
    predicted_labels_probabilities_deque = deque(maxlen = window_size)
 
    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)
 
    while True: 
        status, frame = video_reader.read() 
 
        if not status:
            break
 
        resized_frame = cv2.resize(frame, (image_height, image_width))
         
        normalized_frame = resized_frame / 255

        # load model
        model = tf.keras.models.load_model("Model.h5")
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]
 
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)
 
        if len(predicted_labels_probabilities_deque) == window_size:
 
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)
 
            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
 
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            predicted_class_name = classes_list[predicted_label]
            
            return predicted_class_name
 

if __name__ == "__main__":
    from wsgiref.simple_server import make_server
    with make_server("", 8000, app) as server:
        print("serving on port 8000...")
        server.serve_forever()
    
    