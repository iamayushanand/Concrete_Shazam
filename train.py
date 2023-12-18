import numpy as np
import librosa
import os

from concrete.ml.deployment import FHEModelClient, FHEModelDev, FHEModelServer
from concrete.ml.sklearn import DecisionTreeClassifier

feature_db = []

def compute_features(y):
    feats = []
    win_sz = 22050//2
    for i in range(len(y)//win_sz):
        y_chunk = y[win_sz*i:win_sz*(i+1)]
        mfcc_f = np.sum(librosa.feature.mfcc(y=y_chunk), axis=1)
        feats.append(mfcc_f)
    return feats

def add_features(y, id):
    feats = compute_features(y)
    for feat in feats:
        feature_db.append((feat, id))

def load_dataset():
    for dirpath, dnames, fnames in os.walk("./music_db"):
        idx = []
        for f in fnames:
            #song_name = ' '.join(str(f).split("-")[:-2])
            song_name = f
            y, sr = librosa.load(os.path.join(dirpath, f), duration = 180)
            if song_name not in idx:
                idx.append(song_name)
                print(f"id: {idx.index(song_name)}, song_name: {song_name}")
            add_features(y, idx.index(song_name))
    features = np.array([fp[0] for fp in feature_db])
    ids = np.array([fp[1] for fp in feature_db])
    return features, ids

if __name__ == "__main__":
    # First get some data and train a model.
    X, y = load_dataset()

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    # Train the model and compile it
    model = DecisionTreeClassifier(max_depth=10, n_bits=3)
    model.fit(X, y)
    model.compile(X)
    dev = FHEModelDev("./dev", model)
    dev.save()