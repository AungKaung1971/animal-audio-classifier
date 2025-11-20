Animal Sound Classifier

A machine learning pipeline for classifying animal sounds using audio preprocessing, feature extraction, model training, and evaluation.
This project is designed for hands-on learning, with simple, modular Python scripts and Jupyter notebooks.

📁 Project Structure
animal-sound-classifier/
│
├── data/
│   ├── raw/               # Unprocessed audio files and dataset metadata
│   ├── processed/         # Cleaned & normalized audio
│   └── README.md          # Data sources + preprocessing notes
│
├── notebooks/
│   └── 01-data-exploration.ipynb  # For initial dataset inspection
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py # Audio cleaning, augmentation, feature extraction
│   ├── model.py           # Model definitions (CNN, etc.)
│   ├── train.py           # Training loop
│   ├── evaluate.py        # Evaluation & confusion matrix
│   └── predict.py         # Predict labels for new audio
│
├── models/
│   └── best_model.pth     # Saved PyTorch model
│
├── tests/
│   └── test_data_processing.py    # Unit tests (optional)
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE

🚀 Project Overview

This project aims to build a supervised ML model that can classify animal sounds (e.g., dogs, cats, birds, cows, etc.).
It includes:

Audio preprocessing (trimming, normalization, denoising)

Feature extraction (MFCCs, spectrograms)

Model training using convolutional neural networks

Evaluation with metrics & confusion matrix

Inference on new sound samples

This repo is structured to follow best practices in ML engineering.

🛠️ Setup Instructions
1. Clone the Repository
git clone https://github.com/your-username/animal-sound-classifier.git
cd animal-sound-classifier

2. Create a Virtual Environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

3. Install Dependencies
pip install -r requirements.txt

🎧 Data

Place your audio dataset inside:

data/raw/


After preprocessing, cleaned data will appear in:

data/processed/


You can document your dataset sources inside:

data/README.md

📊 Notebooks

Use Jupyter to explore and understand your data:

jupyter notebook notebooks/01-data-exploration.ipynb

🧩 Code Modules
🔹 data_processing.py

Audio loading

Noise reduction

MFCC extraction

Spectrogram generation

Normalization

Data augmentation

🔹 model.py

CNN architectures for audio classification

Helper functions for building PyTorch models

🔹 train.py

Training loop

Data loaders

Loss functions & optimizers

Model checkpoint saving

🔹 evaluate.py

Accuracy, precision, recall, F1

Confusion matrix visualization

🔹 predict.py

Load trained model

Run inference on new audio file

Output predicted label

🏋️ Training the Model

Example training command:

python src/train.py --epochs 20 --batch-size 32 --lr 0.001


Your best model will be saved in:

/models/best_model.pth

🔍 Running Inference
python src/predict.py --audio path/to/file.wav

🧪 Testing

If using unit tests:

pytest

📄 License

This project is licensed under the MIT License.
Feel free to use, modify, and distribute.

🤝 Contributing

Feel free to open issues or submit pull requests!
This project is designed for personal learning, so improvements are welcome.
