import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import make_embeddings, load_data
from clip_similarity import compute_clip_similarity, prepare_clip_inputs

bert_dim = 768
clip_dim = 512
hidden_dim = 256

# ===============================
# Attention-features for CLIP and BERT Embedding
# ===============================

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
    
# ===============================
# Projection Classifier Model
# ===============================

class ProjectionClassifier(nn.Module):
    def __init__(self, input_dim):
        super(ProjectionClassifier, self).__init__()
        
        # Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: 2 classes
        )
    
    def forward(self, x):
        x = self.projection_head(x)
        x = self.classifier(x)
        return x
    
def load_clip_embedding(file_path):
    data = load_data(file_path, percentage=0.1)
    # Prepare text-image pairs
    texts, images = prepare_clip_inputs(data)

    if texts and images: 
        text_embeddings, _, _, _ = compute_clip_similarity(texts, images)

    return text_embeddings

def text_features(file_path):
    # Example input tensors
    bert_embedding = make_embeddings(file_path)
    clip_text_embedding = load_clip_embedding(file_path)  # CLIP output

    # Initialize fusion module
    fusion_model = AttentionFusion(bert_dim=bert_dim, clip_dim=clip_dim, hidden_dim=hidden_dim)
    fused_output = fusion_model(bert_embedding, clip_text_embedding)
    #print(f"Fused Output Shape: {fused_output.shape}")  # Expected: [1271, 256]

    # Initialize Projection Classifier
    projection_classifier = ProjectionClassifier(input_dim=fused_output.shape[1])
    classification_output = projection_classifier(fused_output)
    #print(f"Text Projection Output Shape: {classification_output.shape}")

    return fused_output #[1271, 256]


def main():
    file_path = "twitter/train.jsonl"
    text_features(file_path)

if __name__ == "__main__":
    main()