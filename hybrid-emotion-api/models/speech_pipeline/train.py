import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa
import os
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# --- ARCHITECTURE ---
class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec2.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values=input_values)
        pooled_output = torch.mean(outputs.last_hidden_state, dim=1) 
        return self.classifier(pooled_output)

# --- DATASET ---
class AudioOnlyDataset(Dataset):
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df['split'] == 'train'].reset_index(drop=True)
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        
        emotions = sorted(self.df['label'].unique())
        self.label2id = {emotion: idx for idx, emotion in enumerate(emotions)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        speech, _ = librosa.load(row['path'], sr=16000)
        input_values = self.processor(
            speech, sampling_rate=16000, return_tensors="pt", 
            padding="max_length", max_length=48000, truncation=True
        ).input_values.squeeze(0)
        
        label = self.label2id[row['label']]
        return input_values, torch.tensor(label, dtype=torch.long)

# --- TRAINING LOOP ---
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Speech-Only Training on: {device.upper()}")

    dataset = AudioOnlyDataset("data/processed/manifest.csv")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SpeechEmotionModel(num_classes=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    os.makedirs("checkpoints/speech", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for input_values, labels in loop:
            input_values, labels = input_values.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(acc=correct/total, loss=loss.item())

        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct / total
        print(f"Epoch {epoch} Summary -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")
        
        torch.save(model.state_dict(), f"checkpoints/speech/speech_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()