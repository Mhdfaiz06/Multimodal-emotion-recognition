import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

# --- ARCHITECTURE ---
class TextEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.classifier = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3), # High dropout prevents it from memorizing sequence lengths
            nn.Linear(256, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output 
        return self.classifier(pooled_output)

# --- DATASET ---
class TextOnlyDataset(Dataset):
    def __init__(self, manifest_path):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df['split'] == 'train'].reset_index(drop=True)
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        
        emotions = sorted(self.df['label'].unique())
        self.label2id = {emotion: idx for idx, emotion in enumerate(emotions)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inputs = self.tokenizer(
            row['text'], return_tensors="pt", 
            padding="max_length", max_length=64, truncation=True
        )
        label = self.label2id[row['label']]
        return inputs.input_ids.squeeze(0), inputs.attention_mask.squeeze(0), torch.tensor(label, dtype=torch.long)

# --- TRAINING LOOP ---
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Text-Only Training on: {device.upper()}")

    dataset = TextOnlyDataset("data/processed/manifest.csv")
    
    # We can use a much larger batch size for text since it takes up less VRAM
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TextEmotionModel(num_classes=7).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    os.makedirs("checkpoints/text", exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for input_ids, attention_mask, labels in loop:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
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
        
        # We save it just for good measure, though we probably won't deploy this specific one!
        torch.save(model.state_dict(), f"checkpoints/text/text_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()