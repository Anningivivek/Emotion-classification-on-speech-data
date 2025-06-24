import streamlit as st
import numpy as np
import librosa
import joblib
import soundfile as sf
from sklearn.preprocessing import LabelEncoder
import os

#loading the model
model = joblib.load("RandomForest_emotion_model.pkl")
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
label_encoder = LabelEncoder()
label_encoder.fit(emotion_labels)

#for the emojis 
emotion_emojis = {
    'angry': '😠',
    'calm': '😌',
    'disgust': '🤢',
    'fearful': '😨',
    'happy': '😄',
    'neutral': '😐',
    'sad': '😢',
    'surprised': '😲'
}


#extracting the features used in the model
def mfcc_values(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc
def delta_values(file_name):
    mfcc = mfcc_values(file_name)
    delta_mfcc =  librosa.feature.delta(mfcc)
    return delta_mfcc
def log_mel_values(file_path, duration=3, offset=0.5, n_mels=128):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel_spec = librosa.power_to_db(mel_spec)
    log_mel_mean = np.mean(log_mel_spec.T, axis=0)
    
    return log_mel_mean
def zcr_values(file_name):
    sig ,sr = librosa.load(file_name,duration=3, offset = 0.5)
    zcr = np.mean(librosa.feature.zero_crossing_rate(sig).T, axis=0)
    return zcr
def spectral_features(file_path, duration=3, offset=0.5):
    y, sr = librosa.load(file_path, duration=duration, offset=offset)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    spectral_features = np.hstack([centroid, bandwidth, rolloff, flatness, contrast])

    return spectral_features
def extract_features(file_name):
    mfcc = mfcc_values(file_name)
    delta = delta_values(file_name)
    log_mel = log_mel_values(file_name)
    zcr = zcr_values(file_name)
    spectral = spectral_features(file_name)
    all_features = np.hstack([mfcc,delta,log_mel,zcr,spectral])
    return all_features
#For streamlit 
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("Speech Emotion Recognition Web App")
st.write("upload the audio_file(.wav)")

uploaded_file = st.file_uploader("Choose an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    # Extract features and make prediction
    try:
        features = extract_features("temp.wav")
        prediction = model.predict([features])[0]
        predicted_emotion = prediction
        st.write(f"Predicted Emotion: **{predicted_emotion.capitalize()}** {emotion_emojis[predicted_emotion]} ")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
