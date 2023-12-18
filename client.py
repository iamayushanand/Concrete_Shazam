"""Client script.

This script does the following:
    - Query crypto-parameters and pre/post-processing parameters
    - Quantize the inputs using the parameters
    - Encrypt data using the crypto-parameters
    - Send the encrypted data to the server (async using grequests)
    - Collect the data and decrypt it
    - De-quantize the decrypted results
"""
import io
import os
from pathlib import Path
import librosa

import grequests
import numpy as np
import requests

from concrete.ml.deployment import FHEModelClient

URL = os.environ.get("URL", f"http://localhost:5000")
STATUS_OK = 200
ROOT = Path(__file__).parent / "client"
ROOT.mkdir(exist_ok=True)

def compute_features(y):
    feats = []
    win_sz = 22050//2
    for i in range(len(y)//win_sz):
        y_chunk = y[win_sz*i:win_sz*(i+1)]
        mfcc_f = np.sum(librosa.feature.mfcc(y=y_chunk), axis=1)        
        feats.append(mfcc_f)
    return feats

if __name__ == "__main__":
    # Get the necessary data for the client
    # client.zip
    y, sr = librosa.load("music_db/DJ Messagroove - Underground Breath.mp3", duration=15)
    feat_sample = compute_features(y)[:30]
    zip_response = requests.get(f"{URL}/get_client")
    assert zip_response.status_code == STATUS_OK
    with open(ROOT / "client.zip", "wb") as file:
        file.write(zip_response.content)

    # Get the data to infer
    X = compute_features(y)
    assert isinstance(X, np.ndarray)

    # Create the client
    client = FHEModelClient(path_dir=str(ROOT.resolve()), key_dir=str((ROOT / "keys").resolve()))

    # The client first need to create the private and evaluation keys.
    client.generate_private_and_evaluation_keys()

    # Get the serialized evaluation keys
    serialized_evaluation_keys = client.get_serialized_evaluation_keys()
    assert isinstance(serialized_evaluation_keys, bytes)

    # Evaluation keys can be quite large files but only have to be shared once with the server.

    # Check the size of the evaluation keys (in MB)
    print(f"Evaluation keys size: {len(serialized_evaluation_keys) / (10**6):.2f} MB")

    # Send this evaluation key to the server (this has to be done only once)
    # send_evaluation_key_to_server(serialized_evaluation_keys)

    # Now we have everything for the client to interact with the server

    # We create a loop to send the input to the server and receive the encrypted prediction
    execution_time = []
    encrypted_input = None
    clear_input = None

    # Update all base64 queries encodings with UploadFile
    response = requests.post(
        f"{URL}/add_key", files={"key": io.BytesIO(initial_bytes=serialized_evaluation_keys)}
    )
    assert response.status_code == STATUS_OK
    uid = response.json()["uid"]

    inferences = []
    # Launch the queries
    for i in range(len(X)):
        clear_input = X[i]
        assert isinstance(clear_input, np.ndarray)
        encrypted_input = client.quantize_encrypt_serialize(clear_input)
        assert isinstance(encrypted_input, bytes)

        inferences.append(
            grequests.post(
                f"{URL}/compute",
                files={
                    "model_input": io.BytesIO(encrypted_input),
                },
                data={
                    "uid": uid,
                },
            )
        )

    # Unpack the results
    decrypted_predictions = []
    for result in grequests.map(inferences):
        if result is None:
            raise ValueError("Result is None, probably due to a crash on the server side.")
        assert result.status_code == STATUS_OK

        encrypted_result = result.content
        decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)[0]
        decrypted_predictions.append(decrypted_prediction)
    bincounts = np.bincount(decrypted_predictions)
    class_prediction = np.argmax(bincounts)
    if bincounts[class_prediction] <= 5:
        print(-1)
    else:
        print(class_prediction)