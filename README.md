Animal Audio Classifier (ESC-50) — CNN on Spectrograms (PyTorch)
================================================================

A PyTorch CNN that classifies **animal sounds** from the **ESC-50** dataset, with an additional **not\_animals** class to reduce false positives. The model is trained on **precomputed spectrogram features** saved as .npy files and evaluated using **5-fold cross-validation**.

Overview
--------

*   **Task:** Multi-class audio classification (animal sounds + not\_animals)
    
*   **Approach:** CNN on 2D spectrogram inputs (.npy tensors)
    
*   **Model:** 3× Conv2D + BatchNorm, MaxPool, AdaptiveAvgPool, Dropout, FC head
    
*   **Training:** 5-fold cross-validation + final model saved to disk
    
*   **Result:** ~**80% accuracy** on the selected animal-related classes (varies by split and dataset balance)
    

Classes
-------

This project trains on the following 10 classes:

```text
cat, cow, crow, dog, frog, hen, not_animals, pig, rooster, sheep   `

```

> not\_animals is intended to represent “everything else” (non-animal audio) so the model can learn to reject non-animal samples instead of forcing an animal label.


Model Architecture
------------------

Implemented in models.py as AudioCNN:

*   Conv2D(1→16, 3×3) + BatchNorm + ReLU + MaxPool
    
*   Conv2D(16→32, 3×3) + BatchNorm + ReLU + MaxPool
    
*   Conv2D(32→64, 3×3) + BatchNorm + ReLU
    
*   AdaptiveAvgPool2D → (32, 32)
    
*   Dropout(p=0.3)
    
*   Linear(64\*32\*32 → 128) + ReLU
    
*   Linear(128 → num\_classes)
    

Repository Structure (expected)
-------------------------------

```plain
ANIMAL_AUDIO_CLASSIFIER/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── __pycache__/
│   ├── models.py
│   ├── preprocessing_data.py
│   └── train.py
├── tests/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
 `
```

Your training script expects **precomputed** feature files here:

```plain
data/processed//*.npy
```

Each .npy file should load as a **2D array** (H×W). The dataset loader will convert it to a tensor and add a channel dimension to make it (1, H, W).

Setup
-----

### Requirements

Core dependencies used in the code:

*   Python 3.x
    
*   PyTorch
    
*   NumPy
    
*   scikit-learn
    

Install (example):

```python
pip install torch numpy scikit-learn
```

> If you used librosa / torchaudio for feature extraction, add them too.

Training
--------

Run 5-fold cross-validation training:

```python
python train.py
```

What it does:

*   Loads .npy spectrograms from data/processed/
    
*   Runs **5-fold CV** (KFold(n\_splits=5, shuffle=True, random\_state=42))
    
*   Trains each fold for **10 epochs**
    
*   Reports fold accuracy + average accuracy
    
*   Saves the final fold model weights to:
    

```python
models/best_model.pth
```

Results
-------

Training reports:

*   Accuracy per fold
    
*   Mean accuracy across all folds
    

Example output format:

```plain
Fold 1 Accuracy: XX.XX%  ...  Average Accuracy: XX.XX%  Saved final model → models/best_model.pth
```

Inference (Real-Time WAV Classification)
----------------------------------------

The project includes an end-to-end pipeline to classify external WAV files in real time using the trained model.

**Note:** Your repo currently shows training + model code. If you paste your inference script (WAV → log-mel → model → prediction), I’ll add:

*   exact CLI usage (python predict.py path/to.wav)
    
*   expected preprocessing parameters (sample rate, n\_mels, hop\_length, normalization)
    
*   example outputs (label + confidence)
    

Notes / Future Improvements
---------------------------

Some clean upgrades you can add later (optional):

*   Use **StratifiedKFold** (keeps class balance per fold)
    
*   Add **learning rate scheduling**
    
*   Add **confusion matrix** and per-class precision/recall
    
*   Add simple audio augmentation (time shift, noise, mixup)
    
*   Save **best model per fold** instead of only final fold
    

Credits
-------

*   Dataset: **ESC-50 Environmental Sound Classification Dataset**
    
