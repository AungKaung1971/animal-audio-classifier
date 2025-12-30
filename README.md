# Animal Audio Classifier (ESC-50)

# 

This repository implements an end-to-end **audio classification pipeline** for recognizing animal sounds using classical audio preprocessing and a convolutional neural network (CNN) in PyTorch. The system is built on a filtered subset of the **ESC-50 environmental sound dataset** and includes data organization, feature extraction, model training with cross-validation, evaluation, and inference.

* * *

## Project Overview

# 

The goal of this project is to classify short audio recordings into animal sound categories. Raw audio files are standardized and converted into **log-mel spectrograms**, which are then used as input to a 2D CNN.

To make the task more realistic, the dataset includes both **animal sounds** and a consolidated **non-animal (“not\_animals”)** class composed of environmental and human-generated sounds.

Model performance is evaluated using **5-fold cross-validation**, and validation accuracy is reported per fold and on average.

* * *

## Repository Structure

```plain
animal-audio-classifier/
├── data/
│   ├── raw/                 # Raw .wav files grouped by class
│   └── processed/           # Log-mel spectrograms saved as .npy
├── models/
│   ├── cnn.py               # CNN architecture
│   └── best_model.pth       # Trained model weights
├── train.py                 # Training with 5-fold cross-validation
├── evaluate.py              # Evaluation utilities
├── inference.py             # Inference on new audio
├── requirements.txt
└── README.md

```

## Dataset

# 

-   **Source:** ESC-50 dataset
    
-   **Classes (10 total):**
    
    -   Animals:  
        `dog, rooster, pig, cow, frog, cat, hen, sheep, crow`
        
    -   Non-animal:  
        `not_animals`
        

Audio files are selected using the ESC-50 metadata (`esc50.csv`) and copied into a class-based folder structure under `data/raw/`. The `not_animals` class aggregates several environmental and human sounds (e.g. rain, breathing, coughing) to act as a negative class.

* * *

## Audio Preprocessing

# 

Each `.wav` file undergoes the following preprocessing steps:

1.  **Resampling**
    
    -   All audio is resampled to **16 kHz**
        
2.  **Amplitude Normalization**
    
    -   Peak normalization to ensure consistent amplitude scale
        
3.  **Duration Standardization**
    
    -   Audio is padded or clipped to **5 seconds** (80,000 samples)
        
4.  **Feature Extraction**
    
    -   **Log-mel spectrogram**
        
        -   Number of mel bands: `128`
            
        -   FFT size: `1024`
            
        -   Hop length: `128`
            
    -   Power spectrogram converted to decibel scale using `librosa.power_to_db`
        
5.  **Storage**
    
    -   Each processed sample is saved as a `.npy` file in:
        
        `data/processed/<class_name>/<filename>.npy`
        

This preprocessing ensures consistent input size while preserving time–frequency information relevant for classification.

* * *

## Model Architecture

# 

The classifier is implemented as a **2D CNN** in PyTorch and operates on single-channel spectrogram inputs.

**Architecture summary:**

-   Input shape: `(1, frequency_bins, time_frames)`
    
-   Convolutional feature extractor:
    
    -   3 × Conv2D layers (kernel size 3×3)
        
    -   Batch Normalization after each convolution
        
    -   ReLU activations
        
    -   Max pooling after the first two convolution layers
        
-   **Adaptive Average Pooling**
    
    -   Output size fixed to `32 × 32`
        
    -   Allows robustness to varying time dimensions
        
-   Classification head:
    
    -   Dropout (`p = 0.3`)
        
    -   Fully connected layer with 128 units
        
    -   Output layer producing logits for 10 classes
        

The model outputs raw logits and is trained using cross-entropy loss.

* * *

## Training Procedure

# 

Training is performed using **5-fold cross-validation** to obtain a robust estimate of generalization performance.

**Training configuration:**

-   Validation strategy: 5-fold cross-validation  
    (`KFold(n_splits=5, shuffle=True, random_state=42)`)
    
-   Loss function: CrossEntropyLoss
    
-   Optimizer: Adam
    
-   Learning rate: `0.001`
    
-   Batch size: `16`
    
-   Epochs per fold: `10`
    
-   Metric: Validation accuracy
    

Class distributions are printed before training to provide visibility into dataset balance.

* * *

## Running the Code

### Installation

# 

Install the required dependencies:

`pip install torch numpy librosa pandas scikit-learn`

GPU acceleration is used automatically if CUDA is available.

* * *

### Training

# 

Run 5-fold cross-validation training:

`python train.py`

This command:

-   Loads `.npy` spectrograms from `data/processed/`
    
-   Trains a CNN independently on each fold
    
-   Reports per-fold and average validation accuracy
    
-   Saves the model weights from the **final fold** to:
    

`models/best_model.pth`

* * *

### Evaluation

# 

Validation accuracy is computed on the held-out fold during cross-validation.  
Per-fold accuracy and the average accuracy across all folds are printed after training.

* * *

### Inference

# 

The inference pipeline applies the same preprocessing steps used during training and runs the trained CNN on new audio files.

`python inference.py --audio path/to/audio.wav`

* * *

## Known Limitations and Suggested Improvements

# 

This project is intentionally kept simple and transparent. Notable limitations and potential improvements include:

1.  **Model Saving Strategy**
    
    -   The saved model corresponds to the final cross-validation fold, not necessarily the best-performing fold.
        
    -   Improvement: track and save the model with the highest validation accuracy across folds.
        
2.  **Class Imbalance**
    
    -   The `not_animals` class aggregates multiple sound types, which may introduce imbalance.
        
    -   Improvement: apply class weighting or balanced sampling.
        
3.  **No Data Augmentation**
    
    -   No audio augmentation (e.g. time shifting, noise injection).
        
    -   Improvement: add augmentation to improve robustness.
        
4.  **Limited Training Duration**
    
    -   Each fold is trained for only 10 epochs.
        
    -   Improvement: increase training time or use early stopping and learning rate scheduling.
        
5.  **Limited Evaluation Metrics**
    
    -   Only overall accuracy is reported.
        
    -   Improvement: add confusion matrices and per-class precision, recall, and F1-scores.
        
6.  **Fixed Feature Representation**
    
    -   Uses log-mel spectrograms exclusively.
        
    -   Improvement: compare against MFCCs or experiment with raw waveform models.
        

* * *

## Summary

# 

This project demonstrates a complete and reproducible machine learning workflow for audio classification, including dataset filtering, feature extraction, CNN-based modeling, and cross-validation-based evaluation. The focus is on clarity, correctness, and extensibility rather than performance maximization, making the repository suitable for technical discussion and further development.
