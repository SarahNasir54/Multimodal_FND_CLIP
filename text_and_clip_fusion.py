import torch
import torch.nn as nn
import torch.nn.functional as F
from text_embedding_projection import text_features
from clip_similarity_projection import clip_features

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1271):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerFusion(nn.Module):
    def __init__(self, text_dim=256, clip_dim=256, hidden_dim=256, num_layers=4, num_heads=8, ff_dim=512, num_classes=2):
        super(TransformerFusion, self).__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.clip_proj = nn.Linear(clip_dim, hidden_dim)
        
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    
    def forward(self, text_features, clip_features):
        text_features = self.text_proj(text_features)  # [seq_len, hidden_dim]
        clip_features = self.clip_proj(clip_features)  # [seq_len, hidden_dim]
        
        text_features = self.pos_encoding(text_features)
        clip_features = self.pos_encoding(clip_features)
        
        text_encoded = self.transformer(text_features.unsqueeze(0)).squeeze(0)  # [seq_len, hidden_dim]
        clip_encoded = self.transformer(clip_features.unsqueeze(0)).squeeze(0)  # [seq_len, hidden_dim]
        
        text_pooled = torch.mean(text_encoded, dim=0)  # Global average pooling
        clip_pooled = torch.mean(clip_encoded, dim=0)
        
        fused_features = torch.cat([text_pooled, clip_pooled], dim=-1)
        output = self.fc(fused_features)
        return output



file_path = "twitter/train.jsonl"

text_embeddings = text_features(file_path)
clip_embeddings = clip_features(file_path)

model = TransformerFusion()
output = model(text_embeddings, clip_embeddings)
print(output.shape)  
