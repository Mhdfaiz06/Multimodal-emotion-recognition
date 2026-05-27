#  Multimodal Emotion Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?style=for-the-badge&logo=pytorch" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface" />
  <img src="https://img.shields.io/badge/FastAPI-0.111.0-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker" />
</p>

<p align="center">
  A deep learning system that detects human emotions from <strong>audio speech</strong>, <strong>text transcripts</strong>, or <strong>both simultaneously</strong> using a hybrid cross-modal attention fusion architecture.
</p>

---
This project bridges the gap between research-oriented AI models and scalable software engineering. It is a production-grade, hybrid Multimodal Emotion Recognition system designed to analyze both *what* is being said (textual semantics) and *how* it is being said (acoustic prosody). 

By fusing state-of-the-art transformer models with a robust backend architecture, the system achieves high-accuracy inference while remaining easily deployable.


##  Live Demo

> **Try it instantly — no setup required:**
>
> 🌐 **[https://huggingface.co/spaces/faiz06/hybrid-emotion-detection-ai](https://huggingface.co/spaces/faiz06/hybrid-emotion-detection-ai)**

Upload a `.wav` file, type a sentence, or provide both to get real-time emotion predictions.

---

### Core Architecture
*   **Acoustic Processing:** Leverages **wav2vec2** to extract prosodic features—such as pitch and intensity—from raw audio signals.
*   **Linguistic Processing:** Utilizes **RoBERTa** for deep semantic analysis of the corresponding text.
*   **Fusion Mechanism:** Implements **cross-modal attention** to intelligently weigh and combine auditory and textual signals for the final emotional classification.
*   **Data Pipeline:** Built around the **TESS** (Toronto Emotional Speech Set) dataset, utilizing a structured data manifest to cleanly align audio and text pairs.

---

##  Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Detected Emotions](#-detected-emotions)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Dataset Setup](#-dataset-setup)
- [Training the Models](#-training-the-models)
- [Running the API](#-running-the-api-locally)
- [Docker Deployment](#-docker-deployment)
- [Using the Web UI](#-using-the-web-ui)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)

---

##  Overview

This project builds a **three-pipeline emotion recognition system** trained on the **TESS (Toronto Emotional Speech Set)** dataset. The system intelligently routes inputs through the best available model:

| Input Provided | Model Used |
|---|---|
| Audio only | Speech model (Wav2Vec2) |
| Text only | Text model (RoBERTa) |
| Audio + Text | Hybrid Fusion Engine (Cross-Modal Attention) |

The key finding: the **multimodal fusion model significantly outperforms either unimodal model alone**, because prosodic signals from speech complement semantic signals from text.

---

##  Architecture

### 1. Speech Pipeline — `Wav2Vec2`
- Backbone: `facebook/wav2vec2-base` (pre-trained on LibriSpeech)
- Input: Raw waveform, resampled to 16 kHz, padded/trimmed to 3 seconds (48,000 samples)
- Attention pooling over temporal hidden states → linear classifier

### 2. Text Pipeline — `RoBERTa`
- Backbone: `roberta-base`
- Input: Tokenized transcript (max 64 tokens)
- `[CLS]` token representation → linear classifier

### 3. Hybrid Fusion Engine — `HybridFusionEngine`
- Both encoders loaded simultaneously
- **Bidirectional Cross-Modal Attention:**
  - Speech attends to text hidden states (s2t)
  - Text attends to speech hidden states (t2s)
- Concatenation of `[speech_cls, text_cls, s_attn, t_attn]` → 3072-dim → MLP classifier
- Lower layers of both encoders frozen to prevent OOM; upper layers fine-tuned

---

##  Detected Emotions

The system classifies into **7 emotion categories**:

| Label | Emotion |
|---|---|
| `angry` | Anger |
| `disgust` | Disgust |
| `fear` | Fear |
| `happy` | Happiness |
| `neutral` | Neutral |
| `ps` | Pleasant Surprise |
| `sad` | Sadness |

---

##  Project Structure

```
multimodal-emotion-recognition/
├── app.py                          # FastAPI server (main entry point)
├── index.html                      # Web UI frontend
├── requirements.txt                # Pinned dependencies
├── Dockerfile                      # Container build config
├── upload.py                       # Script to push checkpoints to HF Hub
│
├── models/
│   ├── fusion_pipeline/
│   │   ├── fusion.py               # HybridFusionEngine architecture
│   │   └── train.py                # Fusion model training script
│   ├── speech_pipeline/
│   │   └── train.py                # SpeechEmotionModel + training script
│   └── text_pipeline/
│       └── train.py                # TextEmotionModel + training script
│
├── utils/
│   ├── dataset.py                  # UnifiedMultimodalDataset (PyTorch)
│   └── data_loader.py              # CSV builder from TESS directory
│
├── data/
│   ├── raw/                        # Downloaded TESS .wav files (gitignored)
│   └── processed/
│       └── manifest.csv            # Label manifest for all splits
│
├── checkpoints/                    # Saved model weights (gitignored)
│   ├── speech/
│   ├── text/
│   └── fusion/
│
└── deployment/
    ├── app.py                      # Hugging Face Spaces entry point
    └── inference.py                # Inference utilities
```

---

## ⚡ Quick Start

### Prerequisites

- Python **3.9+**
- CUDA-capable GPU recommended (CPU training is very slow for transformer models)
- ~8 GB VRAM for fusion training; ~4 GB for unimodal pipelines
- Kaggle account (for dataset download)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Mhdfaiz06/Multimodal-emotion-recognition.git
cd Multimodal-emotion-recognition
```

### Step 2 — Create a Virtual Environment

```bash
python3 -m venv emotion_env
source emotion_env/bin/activate        # Linux / macOS
# emotion_env\Scripts\activate         # Windows
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` pins all versions:

```text
torch==2.2.0
torchaudio==2.2.0
transformers==4.40.0
librosa==0.10.1
soundfile==0.12.1
pandas==2.2.2
scikit-learn==1.4.2
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
pydantic==2.7.1
tqdm==4.66.4
numpy<2.0.0
```

> **Note:** If you have a CUDA GPU, install the matching PyTorch CUDA build instead:
> ```bash
> pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
> ```

---

##  Dataset Setup

The project uses the **TESS (Toronto Emotional Speech Set)** dataset from Kaggle.

### Step 1 — Configure Kaggle API

1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings) → **API** → **Create New Token**
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. Set permissions:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Step 2 — Download and Extract

```bash
pip install kaggle
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d data/raw/
```

### Step 3 — Build the Manifest CSV

```bash
python utils/data_loader.py
```

This scans all `.wav` files, extracts emotion labels from filenames, and generates `data/processed/manifest.csv` with columns: `path, text, emotion, label, speaker, split`.

Expected dataset statistics:
- **Total samples:** ~2,800 audio files
- **Speakers:** 2 (OAF = Older Adult Female, YAF = Young Adult Female)
- **Emotions:** 7 classes, balanced

---

##  Training the Models

Models must be trained in this order: **Speech → Text → Fusion**

Create the checkpoints directories first:

```bash
mkdir -p checkpoints/speech checkpoints/text checkpoints/fusion
```

### Train the Speech Model

```bash
python models/speech_pipeline/train.py
```

- Backbone: `facebook/wav2vec2-base`
- Batch size: 16 | Epochs: 10 | LR: 2e-5
- Checkpoints saved to: `checkpoints/speech/speech_epoch_N.pt`
- Expected training time: ~30–60 min on GPU

### Train the Text Model

```bash
python models/text_pipeline/train.py
```

- Backbone: `roberta-base`
- Batch size: 32 | Epochs: 10 | LR: 2e-5
- Checkpoints saved to: `checkpoints/text/text_epoch_N.pt`
- Expected training time: ~10–20 min on GPU

### Train the Fusion Model

```bash
python models/fusion_pipeline/train.py
```

- Architecture: `HybridFusionEngine` (Wav2Vec2 + RoBERTa + Cross-Modal Attention)
- Batch size: 4 (keep small to avoid OOM) | Epochs: 10 | LR: 2e-5
- Mixed-precision training enabled automatically via `torch.cuda.amp`
- Checkpoints saved to: `checkpoints/fusion/hybrid_epoch_N.pt`
- Expected training time: ~60–120 min on GPU

> ** Important:** The training scripts contain a `break` statement for local testing. **Remove the `break` inside the training loop** before running full training in Colab or on a server.

---

##  Running the API Locally

After training, the final checkpoints are loaded by the FastAPI server. The server expects:
- `checkpoints/speech/speech_epoch_10.pt`
- `checkpoints/text/text_epoch_10.pt`
- `checkpoints/fusion/hybrid_epoch_10.pt`

### Start the Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will:
1. Load all three model checkpoints on startup
2. Serve the web UI at `http://localhost:8000`
3. Expose the prediction endpoint at `http://localhost:8000/predict`

### Test the API (curl)

```bash
# Audio + Text (Fusion)
curl -X POST http://localhost:8000/predict \
  -F "audio_file=@sample.wav" \
  -F "text=I am so happy today"

# Audio only
curl -X POST http://localhost:8000/predict \
  -F "audio_file=@sample.wav"

# Text only
curl -X POST http://localhost:8000/predict \
  -F "text=This is absolutely disgusting"
```

### API Response Format

```json
{
  "emotion": "happy",
  "confidence": 0.9421,
  "model_used": "Hybrid Multimodal"
}
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t emotion-recognition .

# Run the container
docker run -p 7860:7860 emotion-recognition
```

The container uses `python:3.9-slim`, installs `libsndfile1` and `ffmpeg` system dependencies, and starts the server on port 7860 via `uvicorn`.

> To use GPU inside Docker, add `--gpus all` to the run command (requires NVIDIA Container Toolkit).

---

##  Using the Web UI

Open your browser and go to `http://localhost:8000` (local) or the [live Hugging Face Space](https://huggingface.co/spaces/faiz06/hybrid-emotion-detection-ai).

The interface supports three input modes:

**Mode 1 — Audio Only**
1. Click "Upload Audio" and select a `.wav` file
2. Click **Predict**
3. The system routes through the Speech model and returns the predicted emotion with confidence

**Mode 2 — Text Only**
1. Type a sentence in the text box
2. Click **Predict**
3. The system routes through the RoBERTa text model

**Mode 3 — Multimodal (Recommended)**
1. Upload a `.wav` file AND type the corresponding transcript
2. Click **Predict**
3. The Hybrid Fusion Engine processes both and returns the most accurate prediction

The UI displays the predicted emotion label and a confidence score.

---

##  Results

| Model | Architecture | Test Accuracy | Weighted F1 |
|---|---|---|---|
| Speech Only | Wav2Vec2 + Attention Pool | ~93% | ~0.93 |
| Text Only | RoBERTa + CLS head | ~45% | ~0.44 |
| **Hybrid Fusion** | **Wav2Vec2 + RoBERTa + Cross-Attn** | **~96%** | **~0.96** |

The low text-only accuracy (~45%) is an **expected and informative result**: TESS transcripts are single words, so the emotion is entirely in the vocal delivery. The word "back" spoken happily and the word "back" spoken angrily are identical tokens to RoBERTa. This motivates the fusion model — the speech encoder supplies the prosodic signal that text alone cannot provide.

---

##  Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large | Reduce `BATCH_SIZE` to 2 in `fusion/train.py` |
| `RuntimeError: stack expects equal size tensors` | Variable audio length | Fixed by padding/trimming to 48,000 samples in Dataset |
| `nan` loss | Exploding gradients | Add `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` |
| Model not loading on startup | Checkpoint path mismatch | Ensure epoch number in filename matches the path in `app.py` |
| `libsndfile` not found | Missing system library | Run `apt-get install libsndfile1` (Linux) or use Docker |
| Slow inference | Running on CPU | Set `device = "cuda"` or use the hosted Spaces demo |

---

##  Links

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/Mhdfaiz06/Multimodal-emotion-recognition.git |
| Live Demo (HF Spaces) | https://huggingface.co/spaces/faiz06/hybrid-emotion-detection-ai |
| TESS Dataset | https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess |
| Wav2Vec2 Model | https://huggingface.co/facebook/wav2vec2-base |
| RoBERTa Model | https://huggingface.co/roberta-base |

---


## Author

**Mohammed Faiz** 
* **Department**: Computer Science  
* **Institution**: Saintgits College of Engineering, Kottayam

---

##  License

This project is submitted as an academic project for IIIT Hydrabad Research teaser program ,from Mohammed Faiz of Saintgits College of Engineering. All pre-trained model weights are subject to their respective licenses (Meta AI / HuggingFace).
