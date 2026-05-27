from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse 
from fastapi.middleware.cors import CORSMiddleware 
import torch
import librosa
import io
from transformers import Wav2Vec2Processor, RobertaTokenizer

# Import your architectures based on your new, clean folder structure
from models.fusion_pipeline.fusion import HybridFusionEngine
from models.speech_pipeline.train import SpeechEmotionModel
from models.text_pipeline.train import TextEmotionModel

app = FastAPI(title="Multimodal Emotion Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global dictionaries to store our loaded models and processors
models = {}
processors = {}

# Emotion mapping (Make sure this matches your training data labels)
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "ps", "sad"]

@app.on_event("startup")
async def load_infrastructure():
    print(f"Booting up API on {device.upper()}...")
    
    # 1. Load Processors (Tokenizers & Audio Extractors)
    processors['audio'] = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    processors['text'] = RobertaTokenizer.from_pretrained("roberta-base")

    # 2. Load Speech-Only Model
    print("Loading Speech Engine...")
    speech_model = SpeechEmotionModel(num_classes=7).to(device)
    speech_model.load_state_dict(torch.load("checkpoints/speech/speech_epoch_10.pt", map_location=device))
    speech_model.eval()
    models['speech'] = speech_model

    # 3. Load Text-Only Model
    print("Loading Text Engine...")
    text_model = TextEmotionModel(num_classes=7).to(device)
    text_model.load_state_dict(torch.load("checkpoints/text/text_epoch_10.pt", map_location=device))
    text_model.eval()
    models['text'] = text_model

    # 4. Load Hybrid Fusion Model
    print("Loading Hybrid Fusion Engine...")
    hybrid_model = HybridFusionEngine(num_classes=7).to(device)
    hybrid_model.load_state_dict(torch.load("checkpoints/fusion/hybrid_epoch_10.pt", map_location=device))
    hybrid_model.eval()
    models['fusion'] = hybrid_model

    print("✅ All systems initialized and ready for inference!")

@app.post("/predict")
async def predict_emotion(
    audio_file: UploadFile = File(None), 
    text: str = Form(None)
):
    # Route 1: Error Catching (User provided nothing)
    if not audio_file and not text:
        raise HTTPException(status_code=400, detail="You must provide either an audio file, text, or both.")

    # --- Processing Functions ---
    def process_audio(file):
        speech, _ = librosa.load(io.BytesIO(file), sr=16000)
        return processors['audio'](
            speech, sampling_rate=16000, return_tensors="pt", 
            padding="max_length", max_length=48000, truncation=True
        ).input_values.to(device)

    def process_text(text_str):
        inputs = processors['text'](
            text_str, return_tensors="pt", 
            padding="max_length", max_length=64, truncation=True
        )
        return inputs.input_ids.to(device), inputs.attention_mask.to(device)

    # --- THE ROUTER ---
    with torch.no_grad():
        # Route 2: Fusion (Both provided)
        if audio_file and text:
            print("Routing to: HYBRID FUSION MODEL")
            audio_bytes = await audio_file.read()
            input_values = process_audio(audio_bytes)
            input_ids, attention_mask = process_text(text)
            
            logits = models['fusion'](input_values, input_ids, attention_mask)
            used_model = "Hybrid Multimodal"

        # Route 3: Speech Only
        elif audio_file and not text:
            print("Routing to: SPEECH-ONLY MODEL")
            audio_bytes = await audio_file.read()
            input_values = process_audio(audio_bytes)
            
            logits = models['speech'](input_values)
            used_model = "Audio Only"

        # Route 4: Text Only
        elif text and not audio_file:
            print("Routing to: TEXT-ONLY MODEL")
            input_ids, attention_mask = process_text(text)
            
            logits = models['text'](input_ids, attention_mask)
            used_model = "Text Only"

    # --- Format Output ---
    prediction = torch.argmax(logits, dim=-1).item()
    confidence = torch.softmax(logits, dim=-1)[0][prediction].item()

    return {
        "emotion": LABELS[prediction],
        "confidence": round(confidence * 100, 2),
        "model_used": used_model
    }