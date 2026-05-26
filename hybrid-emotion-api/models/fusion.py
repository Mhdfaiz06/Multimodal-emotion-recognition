import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, RobertaModel

class CrossModalAttention(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value):
        # query: (B, 1, 768), key_value: (B, Seq, 768)
        attn_out, _ = self.attn(query, key_value, key_value)
        return self.norm(attn_out.squeeze(1) + query.squeeze(1))

class HybridFusionEngine(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        
        # 1. State-of-the-art Backbones
        self.speech_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.text_encoder = RobertaModel.from_pretrained("roberta-base")
        
        # 2. Freeze lower layers (Crucial for preventing OOM errors on standard hardware)
        for param in self.speech_encoder.parameters(): 
            param.requires_grad = False
        for param in self.text_encoder.encoder.layer[:6].parameters(): 
            param.requires_grad = False

        self.speech_pool = nn.Linear(768, 1)  # Attention pooling over time
        
        # 3. Cross-Attention Logic
        self.s2t_attn = CrossModalAttention(dim=768)
        self.t2s_attn = CrossModalAttention(dim=768)
        
        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(768 * 4, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_values, input_ids, attention_mask):
        # Extract Unimodal Features
        speech_hidden = self.speech_encoder(input_values).last_hidden_state # (B, Seq_S, 768)
        text_hidden = self.text_encoder(input_ids, attention_mask).last_hidden_state # (B, Seq_T, 768)
        
        # Pooling
        speech_weights = torch.softmax(self.speech_pool(speech_hidden), dim=1)
        speech_cls = (speech_hidden * speech_weights).sum(dim=1) # (B, 768)
        text_cls = text_hidden[:, 0, :] # [CLS] token (B, 768)
        
        # Cross-Modal Fusion
        s_attn = self.s2t_attn(speech_cls.unsqueeze(1), text_hidden)
        t_attn = self.t2s_attn(text_cls.unsqueeze(1), speech_hidden)
        
        # Concat & Classify
        fused = torch.cat([speech_cls, text_cls, s_attn, t_attn], dim=-1)
        return self.classifier(fused)

if __name__ == "__main__":
    print("Initializing HybridFusionEngine (Downloading weights if first run)...")
    model = HybridFusionEngine()
    
    print("\nSimulating a forward pass with dummy data...")
    dummy_audio = torch.randn(1, 48000)      # Batch Size 1, 3 seconds audio
    dummy_text = torch.randint(0, 1000, (1, 64)) # Batch Size 1, 64 text tokens
    dummy_mask = torch.ones(1, 64)
    
    logits = model(dummy_audio, dummy_text, dummy_mask)
    print(f"Success! Final Logits Shape: {logits.shape}") 
    # Should print: torch.Size([1, 7]) -> 1 sample, 7 emotion classes