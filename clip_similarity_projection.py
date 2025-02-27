import torch
from clip_similarity import compute_clip_similarity, prepare_clip_inputs
from data_loader import load_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# ==========================
# **Normalization Layer**
# ==========================
class Normalization:
    def __init__(self):
        self.running_mean = 0.0
        self.running_std = 1.0
        self.count = 0

    def update(self, similarity_scores):
        if similarity_scores.numel() == 0:  
            return  

        batch_mean = similarity_scores.mean().item()
        batch_std = similarity_scores.std().item()

        self.count += 1
        momentum = 0.1  
        self.running_mean = momentum * batch_mean + (1 - momentum) * self.running_mean
        self.running_std = momentum * batch_std + (1 - momentum) * self.running_std

    def standardize(self, similarity_scores):
        if similarity_scores.numel() == 0:  
            return torch.zeros_like(similarity_scores)

        normalized_sim = (similarity_scores - self.running_mean) / (self.running_std + 1e-8)
        return torch.sigmoid(normalized_sim)

# ==========================
# **Load CLIP Similarity**
# ==========================
def load_clip_similarity(file_path, similarity):
    data = load_data(file_path, percentage=0.3)

    texts, images = prepare_clip_inputs(data)
    _, _, clip_embeddings, clip_similarity = compute_clip_similarity(texts, images)

    clip_sim = clip_similarity.diag()

    similarity.update(clip_sim)
    standardized_sim = similarity.standardize(clip_sim)

    #print(f"CLIP Embeddings Shape: {clip_embeddings.shape}")
    #print(f"Standardized Similarity Shape: {standardized_sim.shape}")

    return clip_embeddings.to(device), standardized_sim.to(device)

# ==========================
# **Main CLIP Features Function**
# ==========================
def clip_features(file_path, device):
    #print("Extracting CLIP Features...")

    similarity_stats = Normalization()
    clip_embeddings, similarity = load_clip_similarity(file_path, similarity_stats)

    #print(f"Final Output Shapes - CLIP Embeddings: {clip_embeddings.shape}, Similarity: {similarity.unsqueeze(1).shape}")

    return clip_embeddings, similarity 

# if __name__ == '__main__':
#     train_file_path = "twitter/train.jsonl"
#     clip_embeddings, similarity = clip_features(train_file_path, device)
