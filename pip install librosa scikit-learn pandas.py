import os
import pandas as pd
import numpy as np
import librosa
from tkinter import filedialog
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#Create a function to extract audio features
def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return np.mean(mfccs)

#Prompt the user to select audio files
root = tk.Tk()
root.withdraw()
audio_files = filedialog.askopenfilenames(title="Select Audio Files")
root.destroy()

if not audio_files:
    print("No files selected. Exiting.")
    exit()


features = []
labels = []

for audio_file in audio_files:
    mfcc = extract_audio_features(audio_file)
    features.append([mfcc])
    if "happy" in audio_file:
        labels.append("happy")
    else:
        labels.append("sad")

# Train a machine learning model
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Select the optimal solution for Aggression detection
# You can use a similar approach to train a model for aggression detection.

#  Integrate Aggression detection into your script

# Print predicted emotions and aggression
results = {"File Name": audio_files, "Predicted Emotion": model.predict(features)}
aggression_results = {}  # Replace with your aggression prediction results

# Printing results
emotion_table = pd.DataFrame(results)
aggression_table = pd.DataFrame(aggression_results)

print("Emotion Prediction:")
print(emotion_table)

print("\nAggression Prediction:")
print(aggression_table)

# Create a smooth line chart for emotion
plt.figure(figsize=(10, 5))
emotion_values = np.where(emotion_table["Predicted Emotion"] == "happy", 1, -1)
time = np.arange(len(audio_files))
plt.plot(time, emotion_values, marker='o')
plt.xticks(time, [os.path.basename(file) for file in audio_files], rotation=45)
plt.xlabel("Audio Files")
plt.ylabel("Emotion (happy=1, sad=-1)")
plt.title("Emotion Prediction Over Time")
plt.show()

# Display the plots
plt.show()
