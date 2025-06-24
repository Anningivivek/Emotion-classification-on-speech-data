# 🎙️ Speech Emotion Recognition (SER) using Machine Learning

This project is a complete pipeline for recognizing emotions from speech audio data using machine learning techniques. The system extracts features from `.wav` files (like MFCCs, Chroma, and Mel spectrograms), trains a classifier (e.g., Random Forest or XGBoost), and predicts the emotion conveyed in speech.

## 📊 Demo
> 🧪 Streamlit app coming soon...

## 🚀 Features
- Audio feature extraction using Librosa (MFCC, Chroma, Spectral Contrast, Tonnetz, Log-Mel)
- Handles imbalanced classes using oversampling
- Trains classification models (Random Forest, XGBoost, etc.)
- Provides model evaluation (confusion matrix, F1-score)
- Supports testing new audio samples via `test.py` script
- Interactive UI planned with Streamlit

## 🗂️ Project Structure
```
📁 MARS project/
├── 📓 speech_emotio_recognition.ipynb  # Main training and evaluation pipeline
├── 🐍 test.py                          # Script to test model with new data
├── 📁 Audio_Song_Actors_*              # Dataset (e.g., RAVDESS)
├── 📁 models/                          # Saved trained models
├── 📁 features/                        # Extracted features (optional cache)
└── README.md                           # Project overview
```

## 🧰 Tech Stack
- Python 3.10+
- Libraries:
  - `librosa`, `numpy`, `pandas`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`
  - `streamlit` (for UI)
- Audio Dataset: [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/speech-emotion-recognition.git
   cd speech-emotion-recognition
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and place the dataset**
   - Download RAVDESS or your dataset.
   - Place it in a folder like `Audio_Song_Actors_01-24`.

## 🧪 Usage

### 🧠 Train the Model
Run the notebook to extract features, train the model, and evaluate performance:
```bash
jupyter notebook speech_emotio_recognition.ipynb
```

### 🔍 Test with New Audio
```bash
python test.py
```
Make sure your test `.wav` file path is correctly specified in `test.py`.

### 🌐 (Optional) Run the Streamlit App
```bash
streamlit run app.py
```

## 📈 Evaluation Criteria
- Confusion matrix
- Accuracy > 80%
- F1-score > 80% per class

## 📌 To-Do
- [x] Basic feature extraction
- [x] Classifier training
- [x] Evaluation metrics
- [ ] Save/load model
- [ ] Streamlit Web UI
- [ ] Hyperparameter tuning

## ✍️ Authors
- Vivek Anningi

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
