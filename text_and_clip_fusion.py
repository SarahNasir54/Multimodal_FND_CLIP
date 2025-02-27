import torch
import torch.nn as nn
import torch.nn.functional as F
from text_embedding_projection import text_features
from image_projection import image_features  
from clip_similarity_projection import clip_features  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# **Positional Encoding for Transformer**
# ===============================

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1271):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim, device=device)
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-torch.log(torch.tensor(10000.0, device=device)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].to(x.device)

# ===============================
# **Transformer-Based Fusion Model**
# ===============================

class TransformerFusion(nn.Module):
    def __init__(self, text_dim=1280, img_dim=1024, clip_dim=3072, hidden_dim=256, num_layers=4, num_heads=8, ff_dim=512, num_classes=2):
        super(TransformerFusion, self).__init__()

        self.text_proj = nn.Linear(text_dim, hidden_dim).to(device)  
        self.image_proj = nn.Linear(img_dim, hidden_dim).to(device)  
        self.clip_proj = nn.Linear(clip_dim, hidden_dim).to(device)  
        self.similarity_proj = nn.Linear(1, hidden_dim).to(device)  

        self.pos_encoding = PositionalEncoding(hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True).to(device)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),  
            nn.ReLU(),
            nn.Linear(256, num_classes)
        ).to(device)


    def forward(self, text_features, img_features, clip_features, similarity_scores):
        text_features = text_features.to(device)
        img_features = img_features.to(device)
        clip_features = clip_features.to(device)
        similarity_scores = similarity_scores.to(device).unsqueeze(1)  

        # print(f"Raw Text Shape: {text_features.shape}")  
        # print(f"Raw Image Shape: {img_features.shape}")  
        # print(f"Raw CLIP Shape: {clip_features.shape}")  
        # print(f"Similarity Scores Shape: {similarity_scores.shape}")  

        text_features = self.text_proj(text_features)
        img_features = self.image_proj(img_features)
        clip_features = self.clip_proj(clip_features)
        similarity_scores = self.similarity_proj(similarity_scores)

        min_size = min(text_features.shape[0], img_features.shape[0], clip_features.shape[0], similarity_scores.shape[0])
        text_features = text_features[:min_size]
        img_features = img_features[:min_size]
        clip_features = clip_features[:min_size]
        similarity_scores = similarity_scores[:min_size]

        #print(f"Final Matching Sizes - Text: {text_features.shape}, Image: {img_features.shape}, CLIP: {clip_features.shape}, Similarity: {similarity_scores.shape}")

        text_features = self.pos_encoding(text_features.unsqueeze(1))  
        img_features = self.pos_encoding(img_features.unsqueeze(1))  
        clip_features = self.pos_encoding(clip_features.unsqueeze(1)) 
        similarity_scores = similarity_scores.squeeze(-1).squeeze(1) 
        similarity_scores = self.pos_encoding(similarity_scores.unsqueeze(1))  


        text_encoded = self.transformer(text_features)  
        img_encoded = self.transformer(img_features)  
        clip_encoded = self.transformer(clip_features)  
        sim_encoded = self.transformer(similarity_scores.squeeze(-1))  

        text_pooled = torch.mean(text_encoded, dim=1)
        img_pooled = torch.mean(img_encoded, dim=1)
        clip_pooled = torch.mean(clip_encoded, dim=1)
        sim_pooled = torch.mean(sim_encoded, dim=1)

        fused_features = torch.cat([text_pooled, img_pooled, clip_pooled, sim_pooled], dim=-1)  
        output = self.fc(fused_features)  

        return output


# def main():
#     print("Starting processing...")

#     file_path = "twitter/train.jsonl"

#     print("Loading Text, Image & CLIP embeddings...")

#     text_embeddings = text_features(file_path, device)  
#     image_embeddings = image_features(file_path, device)  
#     clip_embeddings, similarity_scores = clip_features(file_path, device)  

#     batch_size = min(text_embeddings.shape[0], image_embeddings.shape[0], clip_embeddings.shape[0], similarity_scores.shape[0])
#     text_embeddings = text_embeddings[:batch_size, :]
#     image_embeddings = image_embeddings[:batch_size, :]
#     clip_embeddings = clip_embeddings[:batch_size, :]
#     similarity_scores = similarity_scores[:batch_size].unsqueeze(1)  

#     print(f"Text Embeddings Shape: {text_embeddings.shape}")  
#     print(f"Image Embeddings Shape: {image_embeddings.shape}")  
#     print(f"CLIP Embeddings Shape: {clip_embeddings.shape}")  
#     print(f"Similarity Scores Shape: {similarity_scores.shape}")  

#     model = TransformerFusion().to(device)

#     output = model(text_embeddings, image_embeddings, clip_embeddings, similarity_scores)
#     print(f"Output Shape: {output.shape}")  

# ===============================
# **Run Script**
# ===============================

# if __name__ == "__main__":
#     main()