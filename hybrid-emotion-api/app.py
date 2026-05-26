from fastapi import FastAPI, UploadFile, Form
import torch
import librosa
import io
import sys
import os

# Ensure we can import from our models directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import Wav2Vec2Processor, RobertaTokenizer
from models.fusion import HybridFusionEngine

app = FastAPI(title="Hybrid Emotion Engine")

# 1. Initialize the Engine
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Booting Inference Engine on {device.upper()}...")

# Load Architecture
model = HybridFusionEngine(num_classes=7).to(device)

# Load Processors
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Mapping based on alphabetical sorting of TESS folders
EMOTIONS = {
    0: "Angry", 
    1: "Disgust", 
    2: "Fear", 
    3: "Happy", 
    4: "Neutral", 
    5: "Pleasant Surprise", 
    6: "Sad"
}

# We will load the weights dynamically when the app starts
@app.on_event("startup")
async def load_weights():
    weights_path = "checkpoints/hybrid_epoch_10.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()
        print("Production weights loaded successfully.")
    else:
        print(f"WARNING: Weights not found at {weights_path}. Model is untrained.")

# 2. The Prediction Endpoint
@app.post("/predict")
async def predict_emotion(audio: UploadFile, text: str = Form(...)):
    # Read raw audio bytes
    audio_bytes = await audio.read()
    
    # Preprocess Audio (Resample to 16kHz for wav2vec2)
    speech, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)
    input_values = audio_processor(
        speech, sampling_rate=16000, return_tensors="pt", 
        padding="max_length", max_length=48000, truncation=True
    ).input_values.to(device)
    
    # Preprocess Text (Tokenize for RoBERTa)
    text_inputs = tokenizer(
        text, return_tensors="pt", 
        padding="max_length", max_length=64, truncation=True
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    # 3. Network Forward Pass
    with torch.no_grad():
        logits = model(input_values, input_ids, attention_mask)
        prediction = torch.argmax(logits, dim=-1).item()
        
        # Calculate confidence percentage
        probabilities = torch.softmax(logits, dim=-1)[0]
        confidence = probabilities[prediction].item()

    return {
        "emotion": EMOTIONS[prediction],
        "confidence": f"{confidence * 100:.2f}%",
        "analyzed_text": text
    }