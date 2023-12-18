import numpy as np
import librosa
import os
from numpy.linalg import norm
from scipy.ndimage import maximum_filter
from scipy.ndimage import generate_binary_structure, binary_erosion
from concrete import fhe
from concrete.ml.sklearn import DecisionTreeClassifier
import time

fingerprint_db = []

def detect_peaks(image, T_percentile = 90.0, freq_range = 50):
    neighborhood = generate_binary_structure(2,2)

    local_max = maximum_filter(image, footprint=neighborhood)==image
    background = (image==0)

    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_peaks = local_max ^ eroded_background
    peak_coords = []
    peak_values = []
    for j in range(image.shape[1]):
        for i in range(freq_range):
            if detected_peaks[i][j]:
                peak_values.append(image[i][j])
                peak_coords.append((i, j))

    T = np.percentile(peak_values, T_percentile)
    peak_coords_reduced = []
    for i in range(len(peak_values)):
        if peak_values[i]>T:
            peak_coords_reduced.append(peak_coords[i])
    return peak_coords_reduced

def fingerprint(stft, T_percentile = 90.0, freq_range = 50):
    peaks = detect_peaks(stft, T_percentile = T_percentile, freq_range=freq_range)
    fingerprints = []
    for i in range(len(peaks)-60):
        current = peaks[i]
        p1 = peaks[i+40] 
        p2 = peaks[i+50] 
        p3 = peaks[i+60] 
        fingerprints.append((current[0]*1.0,p1[0]*1.0,(p1[1]-current[1])*0.1))
        fingerprints.append((current[0]*1.0,p2[0]*1.0,(p2[1]-current[1])*0.1))
        fingerprints.append((current[0]*1.0,p3[0]*1.0,(p3[1]-current[1])*0.1))
    return fingerprints

def add_fingerprint(stft, id):
    fingerprints = fingerprint(stft)
    for i in range(len(fingerprints)):
        fp = fingerprints[i]
        fingerprint_db.append((fp, id))  

    
for dirpath, dnames, fnames in os.walk("./music_db"):
    idx = []
    for f in fnames[:10]:
        song_name = f
        y, sr = librosa.load(os.path.join(dirpath, f))
        stft = np.abs(librosa.stft(y))**2
        if song_name not in idx:
            idx.append(song_name)
        print(f"id: {idx.index(song_name)}, song_name: {song_name}")
        add_fingerprint(stft, idx.index(song_name))

print(len(fingerprint_db))

fp_hashes = np.array([fp[0] for fp in fingerprint_db])
fp_id = np.array([fp[1] for fp in fingerprint_db])
print("initial bincount:",np.bincount(fp_id))

model = DecisionTreeClassifier(max_depth=10, n_bits=4).fit(fp_hashes, fp_id)
configuration = fhe.Configuration(show_progress=True)


def test_sample(sample_path, concrete = False):
    y, sr = librosa.load(sample_path, duration=15)
    stft = np.abs(librosa.stft(y))**2
    fp_sample = fingerprint(stft, T_percentile = 92)[:30]
    print("sample fp len:", len(fp_sample))

    print(np.sum(model.predict_proba(fp_sample), axis = 0))
    print(np.argmax(np.bincount(model.predict(fp_sample))))

    if concrete:
        start = time.time()
        model.compile(fp_hashes, configuration)
        end = time.time()
        print(end-start)

        print(len(fp_sample))
        start = time.time()
        #print(np.sum(model.predict_proba(fp_sample, fhe="execute"), axis = 0))
        print(np.bincount(model.predict(fp_sample, fhe="execute")))
        end = time.time()
        print(end-start)


test_sample("sample.mp3", concrete=True)


