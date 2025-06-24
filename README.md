# 🎙️ Speech Emotion Recognition

This project is a complete pipeline for recognizing emotions from speech audio using machine learning. We extract meaningful features from `.wav` files, train a classifier (Random Forest), and predict the emotional state conveyed in speech. A Streamlit web app is also included for real-time audio emotion prediction.

---

## 📚 Dataset

- 📌 Source: [RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song](https://zenodo.org/records/1188976#.XCx-tc9KhQI)
- 🎯 Note: The dataset is **imbalanced** — oversampling is applied to improve model performance for minority classes.

---

## 📊 Exploratory Data Analysis

### 📈 Waveform & Spectrogram
- Visualized both waveform and spectrogram of audio samples for each emotion using Librosa.

### ⏱️ Duration Distribution
- Histogram shows the distribution of audio durations per emotion category.

---

## 🎛️ Feature Extraction

Features extracted using Librosa:
- **MFCCs** (Mel-Frequency Cepstral Coefficients)
- **Delta MFCCs**
- **Log-Mel Spectrogram**
- **Spectral Properties**
- **Zero Crossing Rate (ZCR)**

All features are extracted using a custom function and saved in:  
📄 `features_dataset.csv`

---

## 🤖 Model Training (Random Forest)

- Used **RandomForestClassifier** from Scikit-learn.
- Model trained on balanced data using oversampling.
- Saved model as: `RandomForest_emotion_model.pkl`
- Evaluation metrics (confusion matrix, classification report) are shown in the notebook.

### 📊 Classification Report (on validation data)
```
              precision    recall  f1-score   support

       angry       0.90      0.83      0.86        75
        calm       0.92      0.97      0.95        75
     disgust       0.84      0.97      0.90        75
     fearful       0.91      0.80      0.85        76
       happy       0.96      0.85      0.90        75
     neutral       0.95      0.96      0.95        75
         sad       0.94      0.87      0.90        76
   surprised       0.84      0.97      0.90        75

    accuracy                           0.90       602
   macro avg       0.91      0.90      0.90       602
weighted avg       0.91      0.90      0.90       602
```

---

## 🧪 Testing

- `test.py` allows you to test the saved model with new `.wav` files.
- Just update the file path in the script and run it with Python.

---

## 🌐 Streamlit Web App

- Upload `.wav` audio files
- Get instant emotion predictions on a simple UI
- Built with Streamlit (run using `streamlit run app.py`)

---

## 🗂️ Project Structure
```
📁 speech_emotion_recognition
├── 📓 speech_emotio_recognition.ipynb     # Main pipeline (EDA, training, evaluation)
├── 🐍 test.py                             # Script to test model on custom audio
├── 📁 models/                             # Trained models
├── 📁 features/                           # Extracted features
├── 📄 features_dataset.csv                # CSV with extracted features
├── 📄 RandomForest_emotion_model.pkl      # Saved model
└── 📄 README.md                           # Project overview
```

---

## ✅ Key Highlights
- Handles class imbalance
- Feature engineering with multiple audio representations
- High-performing Random Forest classifier
- Visual EDA & interpretable metrics
- Testable script & Streamlit UI support

---

## 👨‍💻 Author
**Vivek Anningi**

---

## 📄 License
This project is licensed under the MIT License.
