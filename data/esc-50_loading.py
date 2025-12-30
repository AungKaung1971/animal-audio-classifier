# the following is the script to load data from the ESC-50 repo and classify the different animals into the data folder

import os
import csv
import shutil
import pandas as pd

# standardizing paths
master_file = "data/ESC-50-Master"
csv_path = os.path.join(master_file, "meta/esc50.csv")
print(csv_path)
audio = os.path.join(master_file, "audio")
dest_path = os.path.join("data/raw")

df = pd.read_csv(csv_path)
print(f"shape of data frame: {df.shape}")

animals = ["dog", "rooster", "pig", "cow",
           "frog", "cat", "hen", "sheep", "crow"]

not_animals = ["rain", "sea_waves", "pouring_water",
               "thunderstorm", "crying_baby", "sneeezing", "clapping", "water_drops", "breathing", "coughing"]

for animal in animals:
    os.makedirs(f"{dest_path}/{animal}", exist_ok=True)
    files = df[df["category"] == animal]["filename"]
    for fname in files:
        shutil.copy(os.path.join(audio, fname),
                    f"{dest_path}/{animal}/{fname}")

os.makedirs("data/raw/not_animals", exist_ok=True)

for not_animal in not_animals:
    files = df[df["category"] == not_animal]["filename"][:4]
    for fname in files:
        shutil.copy(os.path.join(audio, fname),
                    f"{dest_path}/not_animals/{fname}")

print("Finshed Copying Animal Sounds from ESC-50")


# resetting data/raw
# rm -rf data/raw/*
