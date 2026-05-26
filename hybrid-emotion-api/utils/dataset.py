import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import Wav2Vec2Processor, RobertaTokenizerFast

SAMPLE_RATE = 16000
MAX_AUDIO_LEN = 48000  # 3 seconds at 16kHz
MAX_SEQ_LEN = 64

class UnifiedMultimodalDataset(Dataset):
    def __init__(self, csv_path="data/processed/manifest.csv", split="train"):
        """
        Loads the manifest and initializes the HuggingFace processors.
        """
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        
        # Initialize the specific processors required by our models
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- 1. Process Raw Audio for wav2vec2 ---
        audio, _ = librosa.load(row["path"], sr=SAMPLE_RATE)
        
        # Trim silence to ensure the model focuses on the speech
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Pad or truncate the audio array to a fixed length (MAX_AUDIO_LEN)
        if len(audio) < MAX_AUDIO_LEN:
            audio = np.pad(audio, (0, MAX_AUDIO_LEN - len(audio)))
        else:
            audio = audio[:MAX_AUDIO_LEN]
            
        # The processor normalizes the waveform specifically for wav2vec2
        audio_inputs = self.audio_processor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )

        # --- 2. Process Text for RoBERTa ---
        # The tokenizer converts the raw string into sub-word integer IDs
        text_inputs = self.tokenizer(
            str(row["text"]), 
            max_length=MAX_SEQ_LEN, 
            padding="max_length",
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "input_values": audio_inputs.input_values.squeeze(0),
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(row["label"], dtype=torch.long)
        }

if __name__ == "__main__":
    # Quick test to ensure the dataset loads correctly
    print("Testing UnifiedMultimodalDataset...")
    ds = UnifiedMultimodalDataset(split="train")
    sample = ds[0]
    print(f"Loaded sample 0 successfully.")
    print(f"Audio shape: {sample['input_values'].shape}")
    print(f"Text IDs shape: {sample['input_ids'].shape}")
    print(f"Label: {sample['label']}")