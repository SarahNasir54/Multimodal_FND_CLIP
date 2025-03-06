import pickle
import torch
import numpy as np
import torch.nn as nn
import os
from data_loader import load_data
#from weibo_data_loader import load_data
from clip_similarity import prepare_clip_inputs, compute_clip_similarity

g = 5  
file_path = "twitter/train.jsonl"
data_dir = "twitter"
# file_path = "weibo/train.jsonl"
# data_dir = "weibo"
batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# **Attention-Based Aggregator for CLIP Features**
# ===============================

class AttentionAggregator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=0)  
        return torch.sum(attn_weights * x, dim=0)  

# ===============================
# **Feature Extraction and Concatenation**
# ===============================

def get_image_features(file_path, data_dir, g, device):
    data = load_data(file_path, percentage=0.3) 

    texts, images = prepare_clip_inputs(data)
    clip_image_embeddings = None
    if texts and images:
        _, clip_image_embeddings, _, _ = compute_clip_similarity(texts, images)

    clip_image_embeddings = clip_image_embeddings.to(device)

    all_image_features = []  

    print("Processing images...")

    for entry in data:
        feature_file = os.path.join(data_dir, f'visual_feature/{entry["post_id"]}.pkl')
        if not os.path.exists(feature_file):
            continue  
        
        with open(feature_file, mode='rb') as f:
            image = pickle.load(f).get('feature1', None)
            if image is None:
                continue
            
            image = torch.Tensor(image).to(device)  
            visual_features = torch.mean(image, dim=(1, 2))  
            visual_features = visual_features.unsqueeze(0).repeat(g, 1).to(device)  

        aggregator = AttentionAggregator(512).to(device)
        clip_aggregated = aggregator(clip_image_embeddings)  
        clip_aggregated = clip_aggregated.unsqueeze(0).repeat(g, 1).to(device)  

        concatenated_features = torch.cat([clip_aggregated, visual_features], dim=1)  

        all_image_features.append(concatenated_features)

    stacked_features = torch.cat(all_image_features, dim=0)  
    #print(f" Final Image Features Shape (Stacked): {stacked_features.shape}")

    min_size = min(stacked_features.shape[0], 1271)  
    stacked_features = stacked_features[:min_size]  

    return stacked_features  

def image_features(file_path, device):
    #print("Extracting image features...")
    image_features = get_image_features(file_path, data_dir, g, device)
    #print(f"Image Features Shape: {image_features.shape}")  
    texts, images = prepare_clip_inputs(load_data(file_path, percentage=0.1))
    _, clip_image_embeddings, _, _ = compute_clip_similarity(texts, images)

    clip_image_embeddings = clip_image_embeddings.to(device)

    min_size = min(image_features.shape[0], clip_image_embeddings.shape[0])
    image_features = image_features[:min_size]
    clip_image_embeddings = clip_image_embeddings[:min_size]

    final_concat = torch.cat([image_features, clip_image_embeddings], dim=1)  

    return final_concat 


# def main():
#     print(f"Using device: {device}")
#     file_path = "weibo/train.jsonl"
#     image_embeddings = image_features(file_path, device)

#     if image_embeddings is None:
#         print("image_features() returned None!")
#     else:
#         print(f"image_features() returned shape: {image_embeddings.shape}")

# if __name__ == "__main__":
#     main()
