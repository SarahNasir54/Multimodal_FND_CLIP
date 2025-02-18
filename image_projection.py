import pickle
import torch
import numpy as np
import torch.nn as nn
import os
from data_loader import load_data
from clip_similarity import prepare_clip_inputs, compute_clip_similarity

region_image_embeds = 5
g = 5
file_path = "twitter/train.jsonl"
data_dir = "twitter"
batch_size = 64

# ===============================
# Attention-Based Aggregator for CLIP Features
# ===============================

class AttentionAggregator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)

    def forward(self, x):
        attn_weights = torch.softmax(self.attention(x), dim=0)  # Compute attention scores
        return torch.sum(attn_weights * x, dim=0)  # Weighted sum of features
    
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

class ImageFeatureProjector(nn.Module):
    def __init__(self, input_dim=4608, output_dim=256):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)  

    def forward(self, x):
        return self.projection(x)

# ===============================
# Feature Extraction and Concatenation
# ===============================

def get_image_features(file_path, data_dir, g, region_image_embeds):
    data = load_data(file_path, percentage=0.1)

    for entry in data:
        #Global visual features
        feature_file = os.path.join(data_dir, 'visual_feature/{}.pkl'.format(entry["post_id"]))
        if not os.path.exists(feature_file):
            continue
        with open(feature_file, mode='rb') as f:
            image = pickle.load(f)['feature1']
            image = torch.Tensor(image)
            visual_features = torch.mean(image, dim=(1, 2))  # [2048]
            visual_features = visual_features.unsqueeze(0).repeat(g, 1)  # Shape: [5, 2048]

        #Entity level features
        feature_file = os.path.join(data_dir, 'region_features/{}.pkl'.format(entry["image"]))
        if not os.path.exists(feature_file):
            continue
        with open(feature_file, mode='rb') as f:
            region_image_feature = pickle.load(f)['features']
            region_feature_length = region_image_feature.shape[0]
            if region_feature_length < 20:
                region_image_feature = np.vstack([region_image_feature, np.zeros((20 - region_feature_length, 2048))])
            region_image_feature = torch.Tensor(region_image_feature)[:region_image_embeds, :]
 

        concatenated_tensor = torch.cat([visual_features, region_image_feature], dim=0)
    
    #Clip image embedding
    texts, images = prepare_clip_inputs(data)

    if texts and images: 
        _, clip_image_embeddings, _, _ = compute_clip_similarity(texts, images)
        #print("Clip Image embedding shape:", clip_image_embeddings.shape)

    aggregator = AttentionAggregator(512)
    clip_aggregated = aggregator(clip_image_embeddings)  # Shape: [512]
    clip_aggregated = clip_aggregated.unsqueeze(0).repeat(g, 1)  # [5, 512]
    #print("clip aggregated features:", clip_aggregated.shape)

    concatenated_features = torch.cat([clip_aggregated, visual_features, region_image_feature], dim=1)  # [5, 4608]

    projector = ImageFeatureProjector(input_dim=4608, output_dim=256)
    projected_features = projector(concatenated_features)

    return projected_features


def image_features(file_path):
    image_features = get_image_features(file_path, data_dir, g, region_image_embeds)
    print("Image features:", image_features.shape)
    if image_features is not None:
        feature_dim = image_features.shape[1]  # Get the feature dimension

        image_projection = ProjectionClassifier(input_dim=feature_dim)

        # Forward pass
        output = image_projection(image_features)
        print("Image Projection Output Shape:", output.shape)  # Expected: [batch_size, 2]
    else:
        print("No valid image features found.")

def main():
    file_path = "twitter/train.jsonl"
    image_features(file_path)

if __name__ == "__main__":
    main()

