import torch
import torch.nn as nn
#import torch.optim as optim
import torch.nn.functional as F
#from torch.utils.data import DataLoader
#import numpy as np
from clip_similarity import compute_clip_similarity, prepare_clip_inputs
from data_loader import load_data

batch_size = 64

# ==========================
# Projection and classifier
# ==========================
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, output_dim=64):
        super(ProjectionHead, self).__init__()
        # Text Projection
        self.fc1_text = nn.Linear(input_dim, 256)
        self.fc2_text = nn.Linear(256, 64)

        # Image Projection
        self.fc1_image = nn.Linear(input_dim, 256)
        self.fc2_image = nn.Linear(256, 64)

    def forward(self, x):
        text_features = F.relu(self.fc1_text(x[:, :512]))  # First 512 = text
        text_features = F.relu(self.fc2_text(text_features))

        image_features = F.relu(self.fc1_image(x[:, 512:]))  # Last 512 = image
        image_features = F.relu(self.fc2_image(image_features))

        return text_features + image_features

class Classifier(nn.Module):
    def __init__(self, input_dim=64, output_dim=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)   # Hidden size 64
        self.fc2 = nn.Linear(64, output_dim)  # Output size 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
    
# ==========================
# Normalization Layer
# ==========================
class Normalization:
    def __init__(self):
        self.running_mean = 0.0
        self.running_std = 1.0
        self.count = 0

    def update(self, similarity_scores):
        """Update running mean and standard deviation."""
        batch_mean = similarity_scores.mean().item()
        batch_std = similarity_scores.std().item()

        self.count += 1
        momentum = 0.1  # Decay factor
        self.running_mean = momentum * batch_mean + (1 - momentum) * self.running_mean
        self.running_std = momentum * batch_std + (1 - momentum) * self.running_std

    def standardize(self, similarity_scores):
        """Apply normalization: (sim - mean) / std, then apply Sigmoid."""
        normalized_sim = (similarity_scores - self.running_mean) / (self.running_std + 1e-8)
        return torch.sigmoid(normalized_sim)
    

def load_clip_similarity(file_path, similarity):
    data = load_data(file_path, percentage=0.1)
    # Prepare text-image pairs
    texts, images = prepare_clip_inputs(data)

    if texts and images: 
        _, _, clip_embeddings, clip_similarity = compute_clip_similarity(texts, images)
        print("Clip Embeddings Shape:", clip_embeddings.shape)

        # Extract self-similarity
        clip_sim = clip_similarity.diag()

        similarity.update(clip_sim)
        standardized_sim = similarity.standardize(clip_sim)

        print("Clip Embeddings Shape:", clip_embeddings.shape)
        print("Standardized Similarity Shape:", standardized_sim.shape)

    return clip_embeddings, standardized_sim

def apply_projection(clip_embeddings, standardized_sim, projection_head, classifier):
    projected_embeddings = projection_head(clip_embeddings)  

    # Reshape similarity for multiplication
    standardized_sim = standardized_sim.unsqueeze(-1)
    # Element-wise multiplication
    fused_features = projected_embeddings * standardized_sim 
    output = classifier(fused_features)

    return output

def main(file_path):
    projection_head = ProjectionHead(input_dim=512)  
    classifier = Classifier(input_dim=64)  

    similarity_stats = Normalization()
    clip_embeddings, similarity = load_clip_similarity(file_path, similarity_stats)

    # Apply projection and classification
    output = apply_projection(clip_embeddings, similarity, projection_head, classifier)
    
    print("Final Output Shape:", output.shape)
    print(output)

if __name__ == '__main__':
    train_file_path = "twitter/train.jsonl" 
    main(train_file_path)