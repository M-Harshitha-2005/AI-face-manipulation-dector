# 🕵️ AI Face Manipulation Detector

A Streamlit web app that detects AI-manipulated (deepfake) faces in **images and videos**. It combines a custom-trained **MesoNet (Meso4)** CNN with a pretrained **Hugging Face deepfake classifier**, and uses **Grad-CAM** heatmap analysis as a third, independent signal — then fuses all three into a single verdict with confidence score.

Includes user authentication and per-user detection history, all backed by a lightweight SQLite database.

---

## ✨ Features

- 🖼️ **Image Detection** — upload a photo, detect the face, and classify it as Real or Fake
- 🎥 **Video Detection** — frame-by-frame face analysis with a running real/fake verdict
- 🔬 **Grad-CAM Visualization** — heatmap overlay showing which facial regions influenced the model's decision, plus a blur/frequency-based authenticity score
- 🧠 **Ensemble Scoring** — combines MesoNet, the Hugging Face model, and Grad-CAM analysis (with override rules for high-confidence fake signals) into one final prediction
- 📊 **Confidence Charts** — bar charts (images) and frame-wise trend charts (video) via Matplotlib
- 🔐 **Login / Signup** — simple username + password authentication
- 📜 **History Tab** — view your past uploads, predictions, and confidence scores

---

## 🧩 How It Works

1. **Face Detection** — OpenCV Haar Cascade locates the face in the uploaded image/frame
2. **Model 1: MesoNet (Meso4)** — a compact CNN (`Meso4_DF.h5`) trained specifically for deepfake artifact detection, outputs a real/fake probability
3. **Model 2: Hugging Face Pipeline** — [`prithivMLmods/Deep-Fake-Detector-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model) provides a second, independent probability
4. **Grad-CAM Analysis** — a gradient-based heatmap is generated from MesoNet's convolutional layers; its color distribution, blur (Laplacian variance), and frequency-domain characteristics (FFT) are analyzed as an auxiliary authenticity signal
5. **Fusion** — the three scores are weighted and combined, with override rules for cases where Grad-CAM strongly indicates manipulation, to produce the final **Real / Fake** verdict and confidence score

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Deep Learning | TensorFlow / Keras, PyTorch, Hugging Face `transformers` |
| Computer Vision | OpenCV, Pillow |
| Visualization | Matplotlib |
| Storage | SQLite (users & detection history) |

---

## 📁 Project Structure

```
AI-face-manipulation-dector/
├── app.py              # Main Streamlit application
├── Meso4_DF.h5          # Pretrained MesoNet weights
├── requirements.txt     # Python dependencies
├── database.db          # SQLite DB (users + history) - created/used at runtime
├── users.db              # legacy/auxiliary user DB
└── uploads/              # Uploaded images saved for history (created at runtime)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9–3.11 recommended
- pip

### Installation

```bash
git clone https://github.com/<your-username>/AI-face-manipulation-dector.git
cd AI-face-manipulation-dector

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. Sign up for an account, log in, then head to **Image Detection** or **Video Detection** from the sidebar.

---

## 📸 Usage

1. **Sign up / Log in** from the landing screen
2. Choose a mode from the sidebar:
   - **🖼️ Image Detection** — upload a `.jpg`/`.jpeg`/`.png` with a visible face
   - **🎥 Video Detection** — upload a `.mp4`/`.avi`/`.mov`; faces are analyzed frame by frame
3. Review the model breakdown (MesoNet, Hugging Face, Grad-CAM), the confidence chart, and the final verdict
4. Check the **📜 History** tab anytime to revisit past results

---

## ⚠️ Limitations & Disclaimer

- Face detection relies on Haar Cascades, which can miss faces at extreme angles, low light, or small sizes.
- MesoNet was trained on a specific deepfake dataset distribution and may not generalize to all manipulation techniques (e.g., diffusion-based face swaps, GAN upscaling).
- This tool is intended for **educational and research purposes only** and should **not** be used as the sole basis for legal, journalistic, or high-stakes decisions about media authenticity.
- Passwords are currently stored in plaintext in SQLite — **do not deploy this as-is with real user credentials** without adding hashing (e.g., `bcrypt`) and proper secrets management.

---

## 🗺️ Roadmap Ideas

- [ ] Hash passwords and add session-token based auth
- [ ] Swap Haar Cascade for a more robust face detector (e.g., MTCNN/RetinaFace)
- [ ] Add support for batch image uploads
- [ ] Export detection history as CSV/PDF report
- [ ] Dockerfile for one-command deployment

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open a PR or an issue.

## 📄 License

This project is available under the MIT License. See `LICENSE` for details (add one if not already present).

## 🙏 Acknowledgements

- [MesoNet](https://github.com/DariusAf/MesoNet) — deepfake detection architecture
- [prithivMLmods/Deep-Fake-Detector-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model) on Hugging Face
- OpenCV Haar Cascades for face detection
