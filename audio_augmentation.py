import librosa
import os
import numpy as np
import soundfile as sf

MUSIC_DB = "./music_db"
OUTPUT_DIR = "./"
NUM_SONGS = 1
SAMPLE_RATE = 22050
NOISE_DEV = 0.01

count = 0
for dirpath, dnames, fnames in os.walk(MUSIC_DB):
    if count >= NUM_SONGS:
        break
    for f in fnames:
        try:
            song_name =  f
            song_path = os.path.join(dirpath, f)
            y, sr = librosa.load(song_path, sr = SAMPLE_RATE, duration = 180)
            noise = np.random.normal(0, NOISE_DEV, y.shape[0])
            y = y+noise
            sf.write(f"{OUTPUT_DIR}/{song_name}", y, sr)
            count += 1
            if count >= NUM_SONGS:
                break
        except:
            pass
