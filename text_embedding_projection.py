import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import make_embeddings
from clip_similarity import compute_clip_text

bert_dim = 768
clip_dim = 512
hidden_dim = 256

class AttentionFusion(nn.Module):
    def __init__(self, bert_dim, clip_dim, hidden_dim):
        super(AttentionFusion, self).__init__()
        
        self.bert_proj = nn.Linear(bert_dim, hidden_dim)  # [1, 768] → [1, 256]
        self.clip_proj = nn.Linear(clip_dim, hidden_dim)  # [1271, 512] → [1271, 256]
        
        # Attention scoring layers
        self.attention_bert = nn.Linear(hidden_dim, 1)  
        self.attention_clip = nn.Linear(hidden_dim, 1)
        
        self.fusion_proj = nn.Linear(hidden_dim, hidden_dim)  # [1271, 256] → [1271, 256]

    def forward(self, bert_embedding, clip_embedding):
        
        bert_proj = self.bert_proj(bert_embedding)  # Shape: [1, 256]
        clip_proj = self.clip_proj(clip_embedding)  # Shape: [1271, 256]

        # Expand BERT embedding to match CLIP sequence length
        bert_proj_expanded = bert_proj.expand(clip_proj.shape[0], -1)  # [1271, 256]

        # Compute attention scores
        att_bert = self.attention_bert(bert_proj_expanded)  # [1271, 1]
        att_clip = self.attention_clip(clip_proj)           # [1271, 1]

        # Normalize with softmax
        att_weights = F.softmax(torch.cat([att_bert, att_clip], dim=1), dim=1)  # [1271, 2]
        att_bert_weight, att_clip_weight = att_weights[:, 0:1], att_weights[:, 1:2]  # [1271, 1]

        # Weighted sum of embeddings
        fused_embedding = att_bert_weight * bert_proj_expanded + att_clip_weight * clip_proj  # [1271, 256]

        # Project the fused representation
        fused_embedding = self.fusion_proj(fused_embedding)  # [1271, 256]

        return fused_embedding  # Final shape: [1271, 256]

file_path = "twitter/train.jsonl"
# Example input tensors
bert_embedding = make_embeddings(file_path)
clip_embedding = compute_clip_text(file_path)  # CLIP output

# Initialize fusion module
fusion_model = AttentionFusion()

# Forward pass
fused_output = fusion_model(bert_embedding, clip_embedding)
print(f"Fused Output Shape: {fused_output.shape}")  # Expected: [1271, 256]
