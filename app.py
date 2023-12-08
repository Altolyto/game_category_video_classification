import numpy as np
import streamlit as st
from keras.models import load_model
import cv2
from collections import deque
import os
from tensorflow import keras

# loading the saved model
loaded_model = load_model("model.h5")
TEMP_DIR = "tempdir/"
# Specify the height and width to which each video frame will be resized in our dataset.
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 30
NUM_FEATURES = 2048

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["PUBGM", "Mobile Legends",  "Free Fire","Valorant"]

def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):

    cap = cv2.VideoCapture(path) # membuka file video menggunakan OpenCV.
    frames = []
# Membaca frame satu per satu dari video
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, resize) # Menyesuaikan Ukuran
            frame = frame[:, :, [2, 1, 0]] # Menukar saluran warna dari BGR (blue-green-red) menjadi RGB (red-green-blue)
            frames.append(frame) # Menambahkan frame yang sudah dibuat kedalam frames

# Proses berhenti jika jumlah frame yang telah diambil mencapai max_frames
            if len(frames) == max_frames:
                break
# Setelah selesai membaca semua frame atau mencapai max_frames, video capture di-release
    finally:
        cap.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


feature_extractor = build_feature_extractor()
# creating a function for Prediction
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def predict_video_with_model(model, video_path):
    frames = load_video(video_path)
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = model.predict([frame_features, frame_mask])[0]

    predicted_class = np.argmax(probabilities)

    return predicted_class, frames

def main():
    # giving a title
    st.title('Video Classification Web App')
    # Upload video file
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    if uploaded_file is not None:
        # store the uploaded video locally
        with open(os.path.join(TEMP_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File Uploaded Successfully")

        if st.button('Classify The Video'):
            # Perform Action Recognition on the Test Video.
            predicted_class, _ = predict_video_with_model(loaded_model, os.path.join(TEMP_DIR, uploaded_file.name))
            
            # If you have class_vocab, you can print the predicted class label
            # Assuming class_vocab is the list of class labels used during training
            predicted_label = CLASSES_LIST[predicted_class]
            st.success(f'Predicted Class Label: {predicted_label}')

    else:
        st.text("Please upload a video file")


if __name__ == '__main__':
    main()
