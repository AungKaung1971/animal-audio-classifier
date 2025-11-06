# Data Folder Overview

## Raw Data
- Contains original recordings from FreeSound and xeno-canto.
- Formats vary (.mp3, .wav, .m4a).
- Do not modify these directly.

## Processed Data
- Cleaned and normalized 16kHz mono .wav files.
- Each file is a 3-second clip.

## Classes
- Dog: barking, whining, growling
- Cat: meowing, hissing, purring
- Bird: chirping, tweeting
- Background: environmental or non-animal noise

# Data sources
- ESC-50: https://github.com/karoldvl/ESC-50
- FreeSound clips: search “dog bark”, “cat meow” (CC0 license)
- Preprocessing: convert to 16 kHz mono WAV using ffmpeg
