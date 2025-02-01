import pickle
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
from data_loader import load_data

region_image_embeds = 5
g = 5
file_path = "twitter/train.jsonl"
data_dir = "twitter"
batch_size = 64

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


def get_image_features(file_path, data_dir, g, region_image_embeds):
    data = load_data(file_path, percentage=0.1)

    for entry in data:
        feature_file = os.path.join(data_dir, 'visual_feature/{}.pkl'.format(entry["post_id"]))
        if not os.path.exists(feature_file):
            continue
        with open(feature_file, mode='rb') as f:
            image = pickle.load(f)['feature1']
            image = torch.Tensor(image)
            visual_features = torch.mean(image, dim=(1, 2))  # This will result in a shape of [2048], because we reduce dimensions (7, 7)
            visual_features = visual_features.unsqueeze(0).repeat(g, 1)  # Shape: [g, 2048]

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

    return concatenated_tensor

image_features = get_image_features(file_path, data_dir, g, region_image_embeds)
print(image_features.shape)
if image_features is not None:
    feature_dim = image_features.shape[1]  # Get the feature dimension

    image_projection = ProjectionClassifier(input_dim=feature_dim)

    # Forward pass
    output = image_projection(image_features)
    print("Image Projection Output Shape:", output.shape)  # Expected: [batch_size, 2]
else:
    print("No valid image features found.")

