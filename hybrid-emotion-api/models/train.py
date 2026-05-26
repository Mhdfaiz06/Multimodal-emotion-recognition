import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import sys
import os

# Add the root directory to the system path so we can import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import UnifiedMultimodalDataset
from models.fusion import HybridFusionEngine

# Hyperparameters
BATCH_SIZE = 4  # Keep this very small to prevent Out-Of-Memory (OOM) errors
EPOCHS = 10
LEARNING_RATE = 2e-5

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing Training on: {device.upper()}")

    # 1. Load Data
    print("Loading datasets...")
    train_ds = UnifiedMultimodalDataset(split="train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Load Model
    print("Initializing Hybrid Engine...")
    model = HybridFusionEngine().to(device)
    
    # 3. Optimizer & Loss
    # We only pass parameters that require gradients (remember we froze the lower layers!)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Scaler for Mixed Precision Training (Saves massive amounts of VRAM)
    scaler = torch.cuda.amp.GradScaler()

    # 4. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=True)
        
        for batch in loop:
            # Move tensors to GPU/CPU
            input_values = batch["input_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            
            # Cast operations to mixed precision
            with torch.cuda.amp.autocast():
                logits = model(input_values, input_ids, attention_mask)
                loss = criterion(logits, labels)
            
            # Backpropagation with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate accuracy metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            loop.set_postfix(loss=total_loss/total, acc=correct/total)
            
            # -------------------------------------------------------------
            # EARLY BREAK FOR LOCAL TESTING: 
            # Remove this break statement when training in Colab!
            print("\n[TEST] Successfully completed 1 forward/backward pass. Breaking loop to save time.")
            break 
            # -------------------------------------------------------------
            
        print(f"Epoch {epoch} Summary -> Loss: {total_loss/total:.4f} | Accuracy: {correct/total:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/hybrid_epoch_{epoch}.pt")
        print(f"Checkpoint saved to checkpoints/hybrid_epoch_{epoch}.pt\n")
        
        # EARLY BREAK FOR LOCAL TESTING
        break

if __name__ == "__main__":
    train()