Animal Sound Classifier

A machine learning pipeline for classifying animal sounds using audio preprocessing, feature extraction, model training, and evaluation. This project is designed for hands-on learning, with simple, modular Python scripts and Jupyter notebooks.

📁 Project Structure
animal-sound-classifier/
├── data/
│   ├── raw/               # unprocessed audio files and dataset metadata
│   ├── processed/         # cleaned & normalized audio
│   └── README.md          # data sources + preprocessing notes
│
├── notebooks/
│   └── 01-data-exploration.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py # audio cleaning, augmentation, feature extraction
│   ├── model.py           # model definitions (CNN, etc.)
│   ├── train.py           # training loop
│   ├── evaluate.py        # evaluation & confusion matrix
│   └── predict.py         # inference on new audio
│
├── models/
│   └── best_model.pth
│
├── tests/
│   └── test_data_processing.py
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

🛠️ Setup
1. Clone the Repository
git clone https://github.com/your-username/animal-sound-classifier.git
cd animal-sound-classifier

2. Install Dependencies
pip install -r requirements.txt

🎧 Data

Put raw audio files into:

data/raw/


Preprocessed audio will be written to:

data/processed/


Dataset source info lives in:

data/README.md

📊 Notebooks

To explore the dataset:

jupyter notebook notebooks/01-data-exploration.ipynb

🧩 Code Modules Overview

data_processing.py

audio loading

trimming, normalization

augmentation

MFCC & spectrogram extraction

model.py

CNN model architectures

train.py

training loop

saving checkpoints

evaluate.py

metrics

confusion matrix

predict.py

run inference on new .wav files

🏋️ Training
python src/train.py --epochs 20 --batch-size 32 --lr 0.001

🔍 Inference
python src/predict.py --audio path/to/file.wav

📄 License

MIT License.
