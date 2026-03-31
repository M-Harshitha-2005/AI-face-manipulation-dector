import streamlit as st
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.preprocessing import image
from transformers import pipeline
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import uuid
import os
import sqlite3

# ---------------- DATABASE ----------------
# ---------------- DATABASE ----------------
import sqlite3
import os
import uuid

# Create uploads folder
os.makedirs("uploads", exist_ok=True)

def connect():
    return sqlite3.connect("database.db", timeout=10)

def init_db():
    conn = connect()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        password TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        image_path TEXT,
        prediction TEXT,
        confidence REAL
    )
    """)

    conn.commit()
    conn.close()

# ✅ IMPORTANT: Initialize DB
init_db()

# ---------------- AUTH ----------------

def signup(username, password):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO users (username, password) VALUES (?, ?)",
        (username, password)
    )

    conn.commit()
    conn.close()

def login(username, password):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )

    user = cursor.fetchone()
    conn.close()
    return user

# ---------------- HISTORY ----------------

def save_history(username, uploaded_file, prediction, confidence):
    conn = connect()
    cursor = conn.cursor()

    # ✅ FIX: avoid overwrite
    file_path = os.path.join("uploads", f"{uuid.uuid4()}_{uploaded_file.name}")

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    cursor.execute(
        "INSERT INTO history (username, image_path, prediction, confidence) VALUES (?, ?, ?, ?)",
        (username, file_path, prediction, confidence)
    )

    conn.commit()
    conn.close()

def get_history(username):
    conn = connect()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT image_path, prediction, confidence FROM history WHERE username=?",
        (username,)
    )

    data = cursor.fetchall()
    conn.close()
    return data

st.set_page_config(page_title="NullBox", layout="wide")
os.environ["PORT"] = os.getenv("PORT", "8501")
IMGWIDTH = 256
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

class Classifier:
    def __init__(self):
        self.model = None

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)

class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))
        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)
        return Model(inputs=x, outputs=y)

@st.cache_resource
def load_meso_classifier():
    classifier = Meso4()
    classifier.load("Meso4_DF.h5")
    return classifier

@st.cache_resource
def load_hf_pipeline():
    return pipeline('image-classification', model="prithivMLmods/Deep-Fake-Detector-Model", device=-1)

meso_classifier = load_meso_classifier()
hf_pipeline = load_hf_pipeline()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_and_preprocess_image(img_pil, target_size=(IMGWIDTH, IMGWIDTH)):
    img = img_pil.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, img
#grad cam function
def analyze_gradcam_features(heatmap):
    # Blur detection using Laplacian
    heatmap_uint8 = np.uint8(255 * heatmap)
    laplacian_var = cv2.Laplacian(heatmap_uint8, cv2.CV_64F).var()

    # Normalize blur score (lower variance = more blur)
    blur_score = 1 / (1 + laplacian_var) 

    # Low frequency estimation using FFT
    f = np.fft.fft2(heatmap)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    low_freq = np.mean(magnitude[:10, :10])  # central region
    total = np.mean(magnitude)

    freq_ratio = low_freq / (total + 1e-8)

    # Final GradCAM confidence
    gradcam_fake_prob = (blur_score + freq_ratio) / 2

    return gradcam_fake_prob, blur_score, freq_ratio

def generate_gradcam(model, img_array):
    grad_model = Model(
        [model.inputs],
        [model.layers[8].output, model.output]   # ✅ FIXED
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # now valid

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / (np.max(heatmap) + 1e-8)

    return heatmap 

# ---------------- COLOR BASED GRADCAM ANALYSIS ----------------
def analyze_heatmap_colors(heatmap):
    heatmap_norm = heatmap / (np.max(heatmap) + 1e-8)

    low_activation = heatmap_norm < 0.3
    high_activation = heatmap_norm > 0.6

    low_ratio = np.sum(low_activation) / heatmap_norm.size
    high_ratio = np.sum(high_activation) / heatmap_norm.size

    # 🔥 Improved scoring (balanced)
    gradcam_fake_prob = (low_ratio * 0.7) + ((1 - high_ratio) * 0.3)

    return gradcam_fake_prob, low_ratio, high_ratio
st.markdown("""
    <style>
    @import url('https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css');
    .main {
        background: linear-gradient(to bottom, #1a202c, #2d3748);
        color: white;
        padding: 2rem;
        min-height: 100vh;
    }
    .sidebar .sidebar-content {
        background-color: #2d3748;
        color: white;
    }
    .stButton>button {
        background-color: #4a5568;
        color: white;
        border-radius: 0.375rem;
        padding: 0.5rem 1rem;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #718096;
    }
    .stFileUploader {
        background-color: #4a5568;
        border-radius: 0.375rem;
        padding: 1rem;
    }
    h1, h2, h3 {
        color: #e2e8f0;
        font-weight: bold;
    }
    .result-box {
        background-color: #2d3748;
        color:white;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)
# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# ---------------- LOGIN / SIGNUP ----------------
if not st.session_state.logged_in:
    st.title("🔐 Login / Signup")

    option = st.radio("Choose", ["Login", "Signup"])

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if option == "Signup":
        if st.button("Create Account"):
            signup(username, password)
            st.success("Account created! Please login.")

    if option == "Login":
        if st.button("Login"):
            user = login(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully")
                st.rerun()
            else:
                st.error("Invalid credentials")

    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("AI FACE MANIPULATION DETECTOR")
st.sidebar.title(f"Welcome {st.session_state.username}")

page = st.sidebar.selectbox(
    "Select Mode",
    ["🖼️ Image Detection", "🎥 Video Detection", "📜 History", "🚪 Logout"]
)
if page == "🚪 Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()
elif page == "📜 History":
    st.title("📜 Detection History")

    records = get_history(st.session_state.username)

    if records:
        for r in records:
            # st.write(f"📁 {r[1]} → {r[2]} ({r[3]:.2f})")
            st.image(r[0], width=150)
            st.write(f"Prediction: {r[1]} | Confidence: {r[2]:.2f}")
    else:
        st.info("No history found")
# st.sidebar.title("DeepFake Detector")
# page = st.sidebar.selectbox("Select Mode", ["🖼️ Image Detection", "🎥 Video Detection"])

if page == "🖼️ Image Detection":
    st.title("🖼️ AI  Image Detector")
    st.markdown("Upload an image to detect if it's real or a deepfake using advanced AI models.")

    uploaded_img = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'], key="image_uploader")

    if uploaded_img:
        img = Image.open(uploaded_img).convert("RGB")
        img_np = np.array(img)
        processed_image, processed_image_pil = load_and_preprocess_image(img)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            st.warning("No faces detected. Please upload an image with a visible face.")
        else:
            x, y, w, h = faces[0]
            face_img = img.crop((x, y, x + w, y + h))
            face_array, face_pil = load_and_preprocess_image(face_img)

            meso_pred = meso_classifier.predict(face_array)
            meso_prob = meso_pred[0][0]
            meso_label = 1 if meso_prob > 0.5 else 0

            prithiv_pred = hf_pipeline(face_pil)
            prithiv_prob = prithiv_pred[0]['score']
            prithiv_label = 1 if prithiv_pred[0]['label'].lower() == 'real' else 0

            combined_prob = (meso_prob + prithiv_prob) / 2
            final_label = 1 if combined_prob > 0.5 else 0

            num_to_label = {1: "Real", 0: "Fake"}

            st.image(img_np, caption="Uploaded Image with Detected Face", use_column_width=True)
            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
            st.image(img_np, caption="Face Detection", use_column_width=True)

            st.markdown("### Results")
            st.markdown(f"<div class='result-box'>MesoNet: {num_to_label[meso_label]} (Probability: {meso_prob:.2f})</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>Prithiv Model: {num_to_label[prithiv_label]} (Probability: {prithiv_prob:.2f})</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>Combined: {num_to_label[final_label]} (Probability: {combined_prob:.2f})</div>", unsafe_allow_html=True)
            #--------------------gradcam------------------
            import tensorflow as tf
            heatmap = generate_gradcam(meso_classifier.model,face_array)
            heatmap = cv2.resize(heatmap,(face_pil.size[0],face_pil.size[1]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + np.array(face_pil)
            st.image(superimposed_img.astype("uint8"), caption="🔥 Grad-CAM Visualization")

            # ---------------- GRADCAM ANALYSIS ----------------
            gradcam_prob, low_ratio, high_ratio = analyze_heatmap_colors(heatmap)

            gradcam_label = "Fake" if gradcam_prob > 0.25 else "Real"

            st.markdown("### 🔬 Grad-CAM Analysis")

            st.markdown(f"<div class='result-box'>Blue Ratio: {low_ratio:.2f} | Active Regions: {high_ratio:.2f}</div>",unsafe_allow_html=True)
            gradcam_real_score = 1 - gradcam_prob
            # base weighted score
            final_combined = (meso_prob * 0.4) + (prithiv_prob * 0.4) + (gradcam_real_score * 0.2)
            # 🔥 OVERRIDE RULE (VERY IMPORTANT)
            if gradcam_prob > 0.6:
                final_label_all = "Fake"
                final_combined = gradcam_prob
            elif meso_prob < 0.4 and prithiv_prob < 0.4:
                final_label_all = "Fake"
            else:
                final_label_all = "Real" if final_combined >= 0.5 else "Fake"




            # st.markdown("### 🧠 Final AI Decision")
            # st.markdown(f"<div class='result-box'>Final: {final_label_all} (Confidence: {final_combined:.2f})</div>", unsafe_allow_html=True)
            # st.markdown(f"<div class='result-box'>GradCAM: {gradcam_label} (Confidence: {gradcam_prob:.2f})</div>", unsafe_allow_html=True)
            # ---------------- GRAPH ----------------
# ---------------- PROFESSIONAL GRAPH ----------------
            fig = plt.figure(figsize=(8, 5))

            labels = ["MesoNet", "Prithiv", "GradCAM"]
            values = [meso_prob, prithiv_prob, gradcam_prob]

            bars = plt.bar(labels, values)

            # Titles & labels
            plt.title("AI Model Confidence Analysis", fontsize=16, pad=15)
            plt.xlabel("Detection Models", fontsize=12)
            plt.ylabel("Confidence Score", fontsize=12)

            # Y-axis limits
            plt.ylim(0, 1)

            # Grid (light and clean)
            plt.grid(axis='y', linestyle='--', alpha=0.5)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.02,
                    f"{height:.2f}",
                    ha='center',
                    fontsize=11,
                    fontweight='bold'
                )

            # Remove top & right borders for modern UI
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            plt.tight_layout()

            st.pyplot(fig)
            save_history(
              st.session_state.username,
              uploaded_img,
              num_to_label[final_label],
              float(combined_prob)
)
elif page == "🎥 Video Detection":
    st.title("🎥 DeepFake Video Detector")
    st.markdown("Upload a video to analyze faces for deepfake detection frame by frame.")

    uploaded_vid = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov'], key="video_uploader")

    if uploaded_vid:
        video_path = f"temp_video_{uuid.uuid4()}.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_vid.read())

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        predictions = []
        confidences = []
        frame_count = 0

        st.info("Processing video. This may take some time...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            for (x, y, w, h) in faces:
                face_crop = frame[y:y + h, x:x + w]
                face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                face_array, face_pil_processed = load_and_preprocess_image(face_pil)

                meso_pred = meso_classifier.predict(face_array)
                meso_prob = meso_pred[0][0]
                meso_label = 1 if meso_prob > 0.5 else 0

                prithiv_pred = hf_pipeline(face_pil_processed)
                prithiv_prob = prithiv_pred[0]['score']
                prithiv_label = 1 if prithiv_pred[0]['label'].lower() == 'real' else 0

                combined_prob = (meso_prob + prithiv_prob) / 2
                final_label = 1 if combined_prob > 0.5 else 0

                predictions.append(1 if final_label == 0 else 0)
                confidences.append(combined_prob)

                display_label = "Fake" if final_label == 0 else "Real"
                color = (0, 0, 255) if final_label == 0 else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{display_label} ({combined_prob:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        os.remove(video_path)

        st.success("✅ Video Analysis Complete")

        if predictions:
            avg_prediction = np.mean(predictions)
            final_verdict = "🔴 Likely FAKE" if avg_prediction > 0.5 else "🟢 Most Likely REAL"

            fig, axs = plt.subplots(1, 2, figsize=(14, 5))
            axs[0].plot(predictions, marker='o', color='blue')
            axs[0].set_title("Frame-wise Prediction (0=Real, 1=Fake)")
            axs[0].set_xlabel("Frame")
            axs[0].set_ylabel("Prediction")
            axs[0].grid(True)

            axs[1].plot(confidences, marker='x', color='orange')
            axs[1].set_title("Confidence per Frame")
            axs[1].set_xlabel("Frame")
            axs[1].set_ylabel("Confidence")
            axs[1].grid(True)

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("### 🧠 Final Verdict")
            st.markdown(f"<div class='result-box'>**Result:** {final_verdict}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'>**Fake Probability (avg):** {avg_prediction:.2f}</div>", unsafe_allow_html=True)
        else:
            st.warning("No faces detected in the video.")
            save_history(
             st.session_state.username,
             uploaded_vid,
             final_verdict,
             float(avg_prediction)
)
