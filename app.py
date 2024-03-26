from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import pickle
import base64
import librosa.display
import matplotlib.pyplot as plt  # Importing pyplot from matplotlib
from waitress import serve
import statistics


# Set the backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from io import BytesIO

app = Flask(__name__)

# Set the upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg'}

# Load the pre-trained models
model1 = load_model('model1.h5')
model2 = load_model('model2.h5')

# Global vars
RANDOM_SEED = 1337
SAMPLE_RATE = 32000
SIGNAL_LENGTH = 5  # seconds
SPEC_SHAPE = (48, 128)  # height x width
FMIN = 500
FMAX = 12500
EPOCHS = 50

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load the LABELS
LABELS = load_pickle('LABELS.pkl')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Modify the generate_spectrogram function
def generate_spectrogram(audio_path):
    y, sr = librosa.load(audio_path)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Use a Flask context to avoid Matplotlib warning
    with app.app_context():
        fig, ax = plt.subplots(figsize=(15, 5))
        librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax)
        ax.set_title('Spectrogram')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')

        # Save the plot to a BytesIO object
        img_byte_io = BytesIO()
        plt.savefig(img_byte_io, format='png')
        plt.close(fig)

    # Convert the image to base64 for embedding in HTML
    img_data = base64.b64encode(img_byte_io.getvalue()).decode('utf-8')

    return img_data

def preprocess_audio(file_path):
    # Open the audio file with librosa
    sig, rate = librosa.load(file_path, sr=SAMPLE_RATE)

    sig_splits = []
    for i in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
        split = sig[i:i + int(SIGNAL_LENGTH * SAMPLE_RATE)]

        # End of signal?
        if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
            break

        sig_splits.append(split)

    mel_spec_list = []
    for chunk in sig_splits:
        hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=chunk,
                                                  sr=SAMPLE_RATE,
                                                  n_fft=1024,
                                                  hop_length=hop_length,
                                                  n_mels=SPEC_SHAPE[0],
                                                  fmin=FMIN,
                                                  fmax=FMAX)

        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()
        mel_spec = np.expand_dims(mel_spec, -1)
        mel_spec = np.expand_dims(mel_spec, 0)
        mel_spec_list.append(mel_spec)

    return mel_spec_list

def decode_prediction(prediction):
    idx = prediction.argmax()
    return LABELS[idx]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audioFile' not in request.files:
        return render_template('index.html', prediction="No file part")

    file = request.files['audioFile']

    if file.filename == '':
        return render_template('index.html', prediction="No selected file")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filename = filename.replace('\\', '/')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img_data = generate_spectrogram(file_path)

        mel_specs = preprocess_audio(file_path)
        predictions = []
        dict_bird = {}

        count = 0
        for mel_spec in mel_specs:
            p = 0.5 * model1.predict(mel_spec)[0] + 0.5 * model2.predict(mel_spec)[0]
            print("PRobability = ",p)
            x = np.argmax(p)
            print("Index of max = ",x) #return as key
            print("Max of prob = ",p[x]) #return as value
            
            print("total sum = ",sum(p))
            count = count + 1
            
            if x in dict_bird:
                dict_bird[x]['sum'] += p[x]
                dict_bird[x]['count'] += 1
            else:
                dict_bird[x] = {'sum':p[x],'count':1}


            # #maximum of those and labels of maximum
            # predictions.append(decode_prediction(p))

        for key in dict_bird:
            dict_bird[key]['prob_average'] = dict_bird[key]['sum'] / dict_bird[key]['count']

        print("Count = ",count)

        total_count = sum(entry['count'] for entry in dict_bird.values())

        # Calculate the average count for each entry
        for key, value in dict_bird.items():
            value['average_count'] = value['count'] / total_count

        print("Modified Dictionary:", dict_bird)


        for key in dict_bird:
            probability_average = dict_bird[key]['prob_average']
            avg_count = dict_bird[key]['average_count']
            if((probability_average >= 0.70) or avg_count >= 0.65):
                predictions.append(LABELS[key])



        print("Predictions is ",predictions)



        if predictions:  # Check if predictions list is not empty
            mode_prediction = statistics.mode(predictions)
            # mode_prediction = predictions

            print("Mode prediction:", mode_prediction)

            # Check if mode_prediction is a known bird species
            if mode_prediction in LABELS:
                os.remove(file_path)  # Remove the uploaded file after processing
                return render_template('index.html', prediction=mode_prediction, img_data=img_data)
            else:
                print("Mode prediction not found in LABELS dataset:", mode_prediction)
                os.remove(file_path)  # Remove the uploaded file after processing
                return render_template('index.html', prediction="Bird not found", img_data=img_data)
        else:
            os.remove(file_path)  # Remove the uploaded file if no predictions were made
            return render_template('index.html', prediction="No bird species predicted", img_data=img_data)

    return render_template('index.html', prediction="Invalid file format")




if __name__ == '__main__':
        serve(app, host="0.0.0.0", port=8081)